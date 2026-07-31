# Online Retail Demand Forecasting

## Summary

Price can be used as an operational lever to actively manipulate demand via pricing strategies. 
Historical SKU demand establishes the baseline forecast, while pricing and recent demand signals refine short-term expectations.


## Business Problem

In high volume online retail where daily sales reach thousands of units, managing inventory without accurate demand forecasting quickly turns into a high-stakes balancing act between two operational traps: tied-up capital from overstocking or lost revenue and customer churn from stockouts. Demand forecasting bridges this gap by transforming historical sales data into predictive operational visibility—ensuring inventory levels precisely mirror real market demand while protecting cash flow and optimizing warehouse throughput. 

---

## Business Solution: Segmented Demand Forecasting

To solve the dual risks of overstocking and stockouts across thousands of daily orders, this project built a Segmented Demand Forecasting System.

EDA revealed that historical demand is highly skewed, ranging from 1 to 19k units. A very tiny fraction (**0.1% of the total transactions**); only 500 out of over 504,731K transactions have quantities over 1K. However, the 500 transactions (0.1% of transactions) account for 10.7% of total volume. This is not a simple long tail but a fundamentally different demand regime.

**Next question was whether to model retail demand, wholesale demand, or both?**

```plaintext
| Type | Typical Quantity|
|---:|---:|
| Retail order | 1–20 |
| Small bulk order | 50–200|
| Wholesale order | 200 - 1000|
| Mega bulk orders | 1000–19000|
```

If a forecasting model sees all these together, it will try to learn a single process for multiple demand-generating mechanisms.
That often hurts forecasting. After careful analysis of the sales velocity (quantiles: min = 1, 25% = 1, 50% = 3, 75% = 12, 90% = 24, 95% = 32, 99% = 120, max = 19,152), I decided to use custom segmentation to handle the variance and split the data into seven operational sub-segments instead of treating all inventory the same. This allowed me to separate the stable high-volume tier from the unpredictable mega-bulk orders. It also allows the model to capture clear, specific demand signals for every type of item—from high-volume daily bestsellers to slow-moving long-tail products.

---

## Deliverable Business Value

### 1. Eliminating Stockouts on Bestsellers (High-Vol Commercial)

**Performance:** Reached a 7.13% weekly error rate.

**Value:** Gives purchasing teams precise baseline demand for core revenue drivers. Ensures top sellers remain in stock without holding excessive buffer inventory.

### 2. Protecting Cash Flow from Extreme Outliers (B2B Bulk Routing)

**Strategy:** Identified and isolated B2B bulk orders (up to 19,000 units/day) into a dedicated manual review segment. The Mega Bulk orders are usually non-cyclical, contract-driven, or ad-hoc, which breaks time-series logic. So I flagged them for manual B2B sales pipeline tracking.

**Value:** Prevents rare, massive bulk spikes from corrupting automated reorder points for standard retail items, avoiding massive unintended overstocking.

### 3. Adapting to Price Sensitivity & Promotions (Commercial & Retail)

**Strategy:** Leveraged price-ratio features to capture how customer demand shifts with price changes and discounts.

**Value:** Allows category managers to predict volume surges caused by marketing promotions and pre-stock warehouses accordingly.

### 4. Preventing Over-Ordering on Slow Bestsellers (Micro & Long-Tail)

**Strategy:** Tailored the model using specialized loss metrics (Huber Loss) for low-volume, sparse items.

**Value:** Prevents the system from over-reacting to one-off sales spikes, keeping capital free and warehouse racks clear of dead stock.

"The system achieved 7.1% to 10.8% weekly WMAPE across all core revenue-generating segments (High-Vol Commercial, Wholesale, and Commercial), providing high-confidence baselines for both vendor procurement and daily warehouse fulfillment."

---

## Overall System Performance 

The system achieved 7.1%, 9.5%, and 10.8% weekly WMAPE across the High-Vol Commercial, Core-Wholesale, and Commercial segments, respectively, providing high-confidence baselines for both vendor procurement and daily warehouse fulfillment. The models achieved weekly WMAPE of 13% for Standard and 18.9% for Heavy-Retail. Although higher, 19% error rate is expected in retail demand which has inherently higher noise from impulsive purchases and quick trends. 

Regardless of the segment, this system's forecasting accuracy is within the typical industry WMAPE benchmarks;  10% – 20% for wholesale / commercial, and 20% – 35% for heavy retail / trending SKUs due to impulse buys and trends, implying that the production team can reliably use the models.

However, after doing research again I found that before deployment, the production team would verify the system's bias. 


### Forecast Bias: Is the system over-forecasting or under-forecasting?

While the model achieved strong Weekly WMAPE across all segments (especially commercial and wholesale), bias evaluation revealed a systematic negative bias (-6.53% to -18.99%) **except for the Core Wholesale segment that had a +1.10% bias**, indicating the raw model tends to under-predict sales spikes. 

- **Core-Wholesale: with a 9.51% weekly WMAPE and +1.10% forecast bias** is production ready as it is because errors are minor and cancel out naturally.
- **High-Vol Commercial achieved the best weekly WMAPE of 7.13% but the forecasting bias (-6.53%)** is slightly above the industry benchmak (± 5%), therefore it needs minor adjustment such as applying a light 1.07x multiplier to eliminate stockout risk.
- **Commercial & Retail Segments (-8.58% to -18.9% Bias):** LightGBM's standard loss function smoothed out high-frequency demand spikes, resulting in systematic under-forecasting in fast-moving retail channels. Further calibration and model tuning, such as custom loss function to penalize negative predictions, are needed. 


### How to Fix Under-forecasting Errors

The model predicts lower sales than actually occur for Commercial and Retail segments. The operational impact of this is that the Warehouse runs out of stock, leading to stockouts, delayed shipments, and lost orders. 

**Business Impact:**

- **Understocking Risk:** Inventory systems relying on this raw forecast will consistently order ~6.5% to -19% fewer units than the market actually buys.
- **Customer Churn:** Top-selling commercial items will frequently hit stockout status before the next replenishment cycle arrives.

After researching I found that in production supply chain, the team would address this in two ways:

#### 1. Production Post-Processing (The Operational Multiplier) or Safety Stock Buffer

- In supply chain planning, the cost of a lost sale is usually higher than the cost of holding one extra unit. Category managers handle negative bias by adding an explicit Safety Stock Cushion to the baseline prediction.
- In production supply chain systems, models with stable negative bias are often adjusted using an Operational Safety Multiplier: 
    Final Forecast = Model Output times 1.1079, 
    adjusting thr predictions upwards to recenter the forecast, driving overall bias back down near 0% without hurting the underlying model pattern.

To prevent stockouts in production environments, I would apply a 1.11x Operational Calibration Factor (or explicit Safety Stock buffer) to the predictions, recentering overall bias to ~0% while preserving baseline precision.

#### 2. Fix Negative Bias Inside the Model During Training

In inventory management, the cost of a stockout is usually higher than the cost of holding extra stock. The most direct technical fix is to change how the algorithm views under-prediction versus over-prediction. Because standard loss functions (MSE/MAE) treat missing a sale by 10 units the exact same as over-buying by 10 units, LightGBM defaults to playing it safe on noisy data, pulling predictions down. I would **implement a custom objective function for LightGBM that penalizes under-predictions more heavily than over-predictions**. e.g., explicitly tell LightGBM: *Under-predicting demand is 30% to 50% worse than over-predicting.* The tree splits will naturally shift upward to avoid the higher penalty.


Alternatively, I would **tune objective and eval metrics (Quantile Loss & Huber Loss).** Although I already used Huber Loss for the Micro segment, I didn't tune model parameters. Tuning Huber's alpha parameter (which controls the transition point between MSE and MAE) changes how aggressively the model reacts to right-tail sales spikes. Also standard models predict the mean (50th percentile). For segments with heavy negative bias (like Heavy Retail at -18.9%), I can use **asymmetric quantile regression (objective='quantile') and set alpha=0.55 or alpha=0.60 (predicting the 55th or 60th percentile)**. This forces the model to target a higher point in the demand distribution.

Lastly, I would **engineer "spike-aware" trend and momentum features.** Negative bias often occurs because the model's features lag behind active real-world surges. Adding features that explicitly capture positive velocity gives the trees early warning signals to push predictions up: e.g.,
    - **Short-Term Velocity Ratios:** Features like *Rolling Mean 3 / Rolling Mean 28*. If this ratio is > 1.0, the product is accelerating.
    - **Consecutive High-Demand Days:** A counter tracking how many days in a row sales exceeded the 75th percentile.
    - **Discount Depth Interaction:** Instead of just raw Price ratio, create *Price ratio * Rolling Mean 7* to help the model learn that discounts on already fast-moving items cause non-linear exponential spikes.

---

## Feature Importance: What Drives the Forecast

Feature importance was evaluated using two complementary LightGBM measures:

- **Gain:** How much a feature reduces prediction error when it is used. High gain indicates strong predictive value.
- **Split count:** How often a feature is used to divide observations in the trees. High split count indicates that the feature is frequently useful for refining decisions, but not necessarily that it produces the largest accuracy improvement.

### Key Findings and Business Interpretation

#### Product Popularity Sets the Baseline

Across most segments, historical SKU demand—especially SKU mean quantity—generates substantially more predictive value than pricing features. This suggests that a product’s established popularity and typical sales volume are the primary drivers of expected demand.

Price ratio is frequently used by the models to refine forecasts, but it generally contributes less overall error reduction. In business terms, pricing acts more like a demand modifier than a demand creator: a discount may increase sales, but the largest volume response is likely to come from products that already have strong underlying demand.

This means promotions should be planned around SKU popularity. Discounting a high-velocity product may create a meaningful volume increase and require additional inventory, while discounting a low-demand product may have a smaller effect because the product's underlying demand remains limited.


### Segment-Level Business Interpretation

| Segment | Primary demand driver | Business interpretation |
|---|---|---|
| Retail | SKU mean quantity | Historical product demand sets the baseline; price ratio refines expected demand around that baseline. |
| Commercial | SKU mean quantity | Established sales velocity dominates, with recent demand and price signals helping adjust short-term forecasts. |
| Standard | SKU mean quantity | Typical SKU demand is the main anchor; relative price provides an important secondary adjustment. |
| Micro | Price ratio and SKU mean quantity | Pricing is unusually influential for sparse, low-volume demand and should be considered in promotion and replenishment decisions. |
| High-Vol Commercial | SKU mean quantity | High-volume products are forecast mainly from stable historical demand, with price ratio providing promotional fine-tuning. |
| Core-Wholesale | SKU demand level and variability | Wholesale demand is driven more by each SKU's typical volume, variability, and recent order behavior than by price alone. |


### Operational Implications

1. **Procurement and base inventory:** Use SKU historical demand statistics to establish baseline purchase quantities, warehouse capacity, and base safety stock.
2. **Pricing and promotions:** Use price ratio as a short-term demand adjustment, especially for Micro, Retail, Standard, and High-Vol Commercial items. Promotions should be paired with temporary inventory buffers for high-velocity SKUs.
3. **Wholesale planning:** Monitor demand variability and recent order signals alongside average demand because wholesale purchasing patterns are less dependent on price and more dependent on order behavior.
4. **Model governance:** Track gain and split importance together. A feature that appears frequently in splits may be useful for local refinements without being the largest contributor to overall forecast accuracy.

> **Business takeaway:** The models forecast demand in two layers: historical SKU demand determines the product's expected volume, while price and recent demand signals fine-tune the forecast for current conditions.

---


## Operational Deployment & Recommendations

Based on multi-segment evaluation and bias diagnostics, the system provides clear guidelines for supply chain, procurement, and warehouse operations:

### 1. Procurement & Automated Reordering (Weekly Horizon)

* **Core-Wholesale & High-Vol Commercial (7%–10% Weekly WMAPE):** 
  * **Action:** **Automate baseline purchase orders.** Accuracy is high enough to drive direct electronic data interchange (EDI) vendor ordering and macro safety stock planning.
  * **Calibration:** Core-Wholesale (+1.1% Bias) can be deployed out-of-the-box. High-Vol Commercial requires a light **1.07x bias multiplier** to ensure 95%+ service levels.

* **Standard & Heavy Retail Segments (13%–19% Weekly WMAPE):**
  * **Action:** **Use for directional planning.** High intrinsic retail noise (impulse buys, fast trends) requires combining model baselines with commercial promo calendars. Apply post-processing scaling factors (**1.15x to 1.23x**) to eliminate negative bias.

### 2. Risk Mitigation & Manual B2B Routing

* **Extreme Bulk Orders (Segment 7):**
  * **Action:** **Route to manual review.** B2B orders exceeding standard catalog thresholds (up to 19,000 units/day) are isolated from automated forecasting pipelines. This prevents one-off bulk buys from artificially corrupting consumer reorder thresholds.

### 3. Warehouse & Labor Scheduling (Daily Horizon)

* **Daily WMAPE (23%–40%):** 
    * **Action:** Do not use raw daily predictions as the sole basis for strict hour-by-hour labor scheduling because daily demand is inherently noisy. Instead, aggregate forecasts into rolling 3-day or weekly planning windows to support more reliable staffing decisions.
    * **Historical Demand Patterns:** Visualize historical demand by day of the week to identify recurring high- and low-volume periods. Use these patterns alongside the model forecast to anticipate predictable weekly peaks and troughs—for example, scheduling additional warehouse labor on consistently high-demand days and reducing staffing on historically quieter days.

- **Operational Value:** Combining weekly forecast volumes with historical day-of-week patterns provides a more stable basis for workforce planning while preserving visibility into recurring daily demand cycles.


### 4. Commercial Strategy & Price Elasticity

* **Promotional Fine-Tuning:**
  * **Action:** Category managers should use the **`Price Ratio`** feature to simulate demand response before launching discounts. Because pricing fine-tunes volume rather than altering baseline capacity, promotions must be paired with temporary safety-stock increases on high-velocity items. 

---

## Next Steps

**What I would do next**

- Look at a breakdown by product category to see what is driving that remaining modeling errors
- Optimize the models using custom loss function to penalize under-predictions more.
- Engineer spike-aware trend and momentum features that capture positive velocity to give the trees early warning signals to push predictions up.

---

## Strategy

I used a a bottom-up forecasting approach where I modeled daily demand and then aggregated the predictions weekly. Aggregating errors from daily to weekly dropped the **Weighted MAPE by more than 60 percent for micro, commercial, and core-wholesale segments**. For standard and heavy retail segments, weekly aggregation reduced the WMAPE by over 50 percent. After doing research, I found that daily e-commerce sales fluctuate heavily due to random noise (e.g., lower sales on Fridays, spikes on Mondays, a rainy Tuesday). When you pool these days into a 7-day bucket, the random daily timing variations cancel each other out. Besides, intraday shipping and order latency have an impact of model errors because web traffic and processing delays, frequently shift a Monday night order to a Tuesday morning fulfillment. Weekly aggregation completely erases this exact InvoiceDate timestamp noise and smoots out the WMAPE. It no longer matters if a buyer placed their restock order on a Tuesday or a Thursday; the bulk volume is captured inside the same 7-day window. 

I also found upon research that most online retailers keep daily models but change how they evaluate and use it because procurement, supplier orders, and manufacturing run on weekly cycles. This strategy is best for businesses needing both granular daily insight and accurate weekly planning as it retains daily trend shapes.
  
---

## Segment Labels

    **Micro**: 1 unit

    **Standard**: 2-3 units

    **Heavy-Retail**: 4-12 units
    
    **Commercial**: 13-32 units
    
    **High-Vol-Comm**: 33-120 units

    **Core-Wholesale**: 121-1000 units

    **Mega-Bulk-Tail**:  1001-19152 units to be handled separately / rules-based.

---

## Set Up (for MacOS)

```bash
# clone repo:
git clone https://github.com/machaniG/retail-demand-forecasting.git

# create virtual environement
python -m venv venv
source venv/bin/activate

# install requirements
pip install -r requirements.txt
```