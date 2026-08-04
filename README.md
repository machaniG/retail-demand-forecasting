# Online Retail Demand Forecasting

## Summary

This project builds a LightGBM demand forecasting model for online retail using historical SKU demand, pricing, rolling demand signals, and calendar features.

The model uses **quantile regression with `alpha=0.60`**, targeting the 60th percentile of demand to place greater emphasis on avoiding under-forecasting than a median forecast.

| Forecast horizon | WMAPE | Forecast bias |
|---|---:|---:|
| Weekly | **11.3%** | **-1.8%** |
| Daily | **63.0%** | Not reported |

The weekly forecast is the primary operational output because weekly aggregation reduces the effect of volatile day-to-day ordering patterns.

**Key finding:** Historical SKU demand is the strongest forecast anchor. Price and price ratio are also major drivers, while rolling demand and calendar features refine the forecast based on current momentum and timing.

---

## Business Problem

Online retailers must balance two competing inventory risks:

- **Overstocking:** Excess inventory ties up working capital, increases storage costs, and raises the risk of markdowns or obsolescence.
- **Stockouts:** Insufficient inventory causes lost sales, delayed fulfillment, and poorer customer experience.

Demand varies substantially across products and over time. A forecasting system must estimate expected demand while accounting for product popularity, pricing, recent sales momentum, and recurring calendar patterns.

---

## Business Solution

The project uses a **single LightGBM model** trained across the 99th percentile of fast moving weekly selling SKUs. The model learns demand patterns using a shared feature set that captures:

- Long-term SKU demand characteristics
- Current and relative pricing
- Recent demand momentum and volatility
- Product lifecycle maturity
- Weekly and seasonal calendar effects

The model uses LightGBM's **quantile objective with `alpha=0.60`**. This targets a forecast above the conditional median and is intended to reduce the operational risk of under-forecasting when stockouts are more costly than holding a modest amount of additional inventory.

---

## EDA Insights

EDA revealed that historical demand is highly skewed, ranging from 1 to 19k units. A very tiny fraction (**0.1% of the total transactions**); only 500 out of over 504,731K transactions have quantities over 1K. However, the 500 transactions (0.1% of transactions) account for 10.7% of total volume. This is not a simple long tail but a fundamentally different demand regime.

**Next question was whether to model retail demand, wholesale demand, or both?**


| Type | Typical Quantity|
|---:|---:|
| Retail order | 1–20 |
| Small bulk order | 50–200|
| Wholesale order | 200 - 1000|
| Mega bulk orders | 1000–19000|


If a forecasting model sees all these together, it will try to learn a single process for multiple demand-generating mechanisms.
That often hurts forecasting. After detailed EDA, I found that most SKUs sell from 1 unit to large of batches and that basically all customers place both small and large orders. This means that the data cannot be segmented using either **Customer ID or SKU** alone. 

### Outlier Removal

As an initial step, I classified each SKU by selling frequency and isolated fast moving from slow moving products based on how many distinct months the product recorded a sale. After careful analysis of the sales velocity (quantiles: min = 1, 25% = 1, 50% = 3, 75% = 12, 90% = 24, 95% = 32, 99% = 120, max = 19,152), rows above the 99th percentile (≥120 units) were excluded from training. This allowed me to separate the stable high-volume tier from the unpredictable mega-bulk orders. It also allows the model to capture clear, specific demand signals for every type of item.

---

## Model Performance

### Weekly Forecast Performance

The model achieved a **weekly WMAPE of 11.3%** with a **forecast bias of -1.8%**.

- **WMAPE of 11.29%:** Weekly forecast error is low relative to the total demand volume being forecast.
- **Daily WMAPE: 63%**
- **Operational interpretation:** The weekly forecast can serve as a baseline input for procurement, replenishment, and medium-term inventory planning, subject to normal business review and service-level policies.

The system achieved 11.29% weekly WMAPE on fast moving weekly SKUs using quantile regression, providing high-confidence baselines for both vendor procurement and daily warehouse fulfillment. This falls within typical industry WMAPE benchmarks for wholesale/commercial-leaning demand (commonly cited as **10%–20%**), implying that the inventory team can reliably use the models. Although higher, the 63% daily error rate is expected in retail demand which has inherently higher noise from impulsive purchases and quick trends as long as it smooths out over 7 day window. Daily e-commerce sales fluctuate heavily due to random noise (e.g., lower sales on Fridays, spikes on Mondays, a rainy Tuesday), yielding higher daily WMAPE. When you pool these days into a 7-day bucket, the random daily timing variations cancel each other out. Besides, intraday shipping and order latency have an impact of model errors because web traffic and processing delays, frequently shift a Monday night order to a Tuesday morning fulfillment. Weekly aggregation completely erases this exact InvoiceDate timestamp noise and smoots out the WMAPE. 

However, after doing research again I found that before deployment, the production team would verify the system's bias.


### Forecast Bias: Is the system over-forecasting or under-forecasting?

- **Forecast bias: -1.76%** (a slight, near-negligible under-prediction)

Bias evaluation revealed a slight **near negligible negative bias (-1.76%)**. indicating the model has a small tendency of **under-predicting sales by only 2 percent**, which is within the **industry benchmak (forecast bias ±5%)**. The near-zero bias means the forecast is not systematically skewed in either direction at the aggregate level. The model is production ready as it is because errors are minor and cancel out naturally. However, because the model predicts lower sales than actually occurs by 2 percent, it needs minor adjustment such as applying a light 1.07x multiplier to eliminate stockout risk


---

## Feature Importance: What Drives the Forecast

Feature importance was evaluated using two complementary LightGBM measures:

- **Split count:** How often a feature is used to divide observations in the decision trees. A high split count means the feature is frequently useful for refining predictions.
- **Gain:** The total reduction in the model's loss produced by splits using that feature. A high gain means the feature contributes strongly to improving predictive accuracy.

### Top Features

| Rank | Top 10 by Split | Top 10 by Gain |
|---:|---|---|
| 1 | Price (3,543) | SKU mean quantity (15,032,493) |
| 2 | Price ratio (3,369) | Price (6,040,753) |
| 3 | SKU median quantity (1,950) | Price ratio (5,803,695) |
| 4 | Days since first sale (1,243) | SKU median quantity (1,448,734) |
| 5 | SKU avg price (1,129) | Rolling mean 7 (777,260) |
| 6 | SKU mean quantity (998) | Rolling mean 28 (681,667) |
| 7 | Rolling mean 14 (788) | SKU avg price (597,102) |
| 8 | Rolling std 7 (749) | Day of week (516,581) |
| 9 | Days to Christmas (677) | Rolling mean 14 (371,447) |
| 10 | Day of week (640) | Days since first sale (337,522) |

### Business Interpretation

The feature importance results indicate that the model forecasts demand in two layers.

#### 1. SKU demand establishes the baseline

**SKU mean quantity** is the largest gain contributor by a wide margin, and **SKU median quantity** is also highly important. This indicates that a product's established demand level is the strongest anchor for the forecast.

**Business translation:** Product popularity matters. A SKU with consistently strong historical demand is expected to remain a higher-volume product than a SKU with weak historical demand, all else being equal.

#### 2. Price modifies demand around the SKU baseline

**Price** and **price ratio** are the two most frequently used split features and are also the second- and third-largest contributors by gain.

**Business translation:** Pricing is a major demand lever, but it generally modifies demand around the product's underlying popularity rather than replacing that baseline. A discount may increase sales, but the largest absolute volume response is likely to occur when the discounted SKU already has meaningful demand.

This is a predictive interpretation, not a causal estimate of price elasticity. Feature importance shows how useful pricing is to the model; it does not by itself prove how much a price change causes demand to increase.

#### 3. Recent demand momentum refines the forecast

**Rolling mean 7, rolling mean 14, rolling mean 28, and rolling standard deviation 7** capture recent sales level and short-term volatility.

**Business translation:** After establishing the SKU's long-term demand baseline, the model checks whether demand is currently accelerating, slowing, or becoming more volatile.

#### 4. Calendar and lifecycle features capture timing effects

**Day of week, days to Christmas, and days since first sale** help the model account for recurring weekly patterns, seasonal demand, and product maturity.

**Business translation:** The same SKU may have different expected demand depending on the day of the week, proximity to the holiday period, and how long the product has been active.

### Feature Importance Takeaway

- **Historical SKU demand sets the expected volume, price and relative price adjust demand around that baseline, and recent demand plus calendar signals refine the timing of the forecast.**

---

## Operational Deployment & Recommendations

### 1. Procurement and Replenishment

Use the **weekly forecast** as the primary baseline for purchase planning and replenishment.

Because the model has a small negative bias of **-1.8%**, address the bias before automated reordering. A small, consistent under-prediction means inventory systems relying on the raw forecast will order marginally fewer units than the market buys. Two standard approaches are usually applied:

- **Operational multiplier (fastest to deploy):** apply a small upward correction factor (~1.018x, derived from the bias) to recenter the forecast near zero bias without retraining.
- **Re-tune alpha:** since bias is governed largely by the quantile objective's alpha, a small alpha increase (e.g., 0.62–0.65) may close the remaining gap directly, cheaper to validate than adding new features, since it doesn't change what the model is learning, just where it's calibrated.

Inventory decisions should still incorporate service-level targets, supplier lead times, and safety-stock policies.

### 2. Scope SKUs Tiers Appropriately: Only weekly tier to automated reordering

Only the 2,358 SKUs classified into the weekly forecast tier should feed automated reordering. This tier represents 87% of volume, so the operational coverage is high despite covering just over half the SKU catalog by count.

Route monthly-tier and manual B2B routing appropriately, not by default

- **Monthly tier (32% of SKUs, 12% of volume):** real, recurring demand, just not enough for week-level patterns. Recommend either a lighter monthly-aggregated model (not yet built) or monthly-cadence manual review, rather than forcing these into the weekly pipeline.
- **Manual tier (12% of SKUs, <1% of volume):** too sparse to model statistically. Route to simple reorder-point rules; a trained model would be fitting noise here.
- **Flagged manual-review spikes (within the weekly tier):** large, irregular orders identified during EDA should continue to bypass the automated forecast and go to manual B2B tracking, regardless of which tier the SKU otherwise belongs to.


### 3. Pricing and Promotion Planning

Price and price ratio are major model drivers. Category managers can use pricing scenarios to assess how forecast demand changes under different price levels.

Promotions should be evaluated together with SKU demand history:

- High-demand SKUs may require temporary inventory buffers during promotions.
- Low-demand SKUs may show a smaller absolute volume response even when discounted.
- Promotional forecasts should be monitored against actual lift because feature importance is not a causal elasticity estimate.

Category managers simulating a discount should weight the expected volume lift by the SKU's baseline demand (from feature importance: SKU mean quantity dominates gain), a discount on an already-high-velocity item is far more likely to require a temporary inventory buffer than the same discount on a low-baseline item.


### 4. Warehouse and Labor Scheduling

Do not use the raw daily forecast as the sole basis for rigid hour-by-hour staffing because daily WMAPE is **63%**.

Instead:

- Use weekly forecast totals to establish the overall labor and fulfillment requirement.
- Visualize historical demand by **day of week** to identify recurring high- and low-volume periods.
- Combine these recurring day-of-week patterns with the weekly forecast to distribute expected workload across the week.
- Schedule additional labor on consistently high-demand days and reduce staffing on historically quieter days, while allowing for operational buffers.

This approach uses the model's stronger weekly signal while retaining useful information about recurring daily demand patterns.

### 4. Forecast Monitoring

Monitor the model after deployment using:

- Weekly WMAPE
- Forecast bias
- Error by SKU and product category
- Error during promotions and seasonal periods
- Performance drift over time

Feature importance should also be monitored after retraining. Large changes in the importance of price, SKU demand, or rolling features may indicate changes in customer behavior, pricing strategy, or product mix.

---

## Forecasting Strategy

The model is trained at the daily level and evaluated both daily and after aggregating predictions to weekly totals.

Weekly aggregation improves operational usefulness because it reduces the effect of day-to-day timing variation. Demand that occurs on Tuesday instead of Wednesday may create a large daily error but have little impact on the total weekly volume.

The resulting strategy preserves daily forecast detail while using weekly forecasts as the primary planning output for procurement and inventory decisions.


---

## Limitations & Next Steps

- **Monthly and manual tiers are identified but not modeled.** They are currently scoped out rather than forecast — a reasonable next iteration would build a lightweight monthly-cadence model for the monthly tier specifically, given it still carries a non-trivial 12% of volume.
- **Recurring-bulk orders are classified but not yet used as a feature.** SKUs with a regular large-order cadence were flagged during EDA but a `days_since_last_spike`-style feature has not been added to the trained model. This likely affects a small SKU subset but could recover some of the volume currently excluded by the row-level cutoff.
- **Static SKU features imply the model needs periodic refitting.** Since `SKU mean quantity` (frozen at fit time) dominates gain, the model will be slow to react if a SKU's demand pattern genuinely shifts mid-deployment. A refit cadence (e.g., monthly) should be built into the production plan rather than treating the model as fit-once.
- **Bias, while small, hasn't been corrected in production.** The -1.76% bias should be addressed (via multiplier or alpha re-tuning, above) before this model drives automated ordering.
- **Category-level breakdown of remaining error is unexplored** — a next step would be checking whether residual error clusters by product category, which could point to additional useful features.


---

## Setup (macOS)

```bash
# Clone the repository
git clone https://github.com/machaniG/retail-demand-forecasting.git

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
