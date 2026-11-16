-- Time between a user's sessions, and what that implies for a churn definition.
--
-- A churn threshold picked by intuition ("30 days") is usually wrong for the product it is applied
-- to. The distribution of return gaps gives a defensible one: the percentile beyond which users
-- rarely come back. Reporting the percentiles instead of a single number is the point — the
-- threshold is a business choice, and this gives it a basis.
with gaps as (
    select
        user_id,
        session_index,
        hours_since_previous_session / 24.0 as days_since_previous
    from main_intermediate.int_sessions
    where hours_since_previous_session is not null
)

select
    count(*)                                                        as return_visits,
    count(distinct user_id)                                         as returning_users,
    round(avg(days_since_previous), 2)                              as mean_gap_days,
    round(quantile_cont(days_since_previous, 0.50), 2)              as median_gap_days,
    round(quantile_cont(days_since_previous, 0.75), 2)              as p75_gap_days,
    round(quantile_cont(days_since_previous, 0.90), 2)              as p90_gap_days,
    round(quantile_cont(days_since_previous, 0.95), 2)              as p95_gap_days,
    round(quantile_cont(days_since_previous, 0.99), 2)              as p99_gap_days
from gaps
