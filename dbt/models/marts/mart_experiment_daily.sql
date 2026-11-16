-- Cumulative experiment metrics by day and variant: the readout curve.
--
-- Cumulative rather than daily, because that is how an experiment is actually watched, and because
-- it makes the point about peeking visible: the curve wanders early and settles late, so a rule
-- that stops at the first crossing is reading noise. The dashboard draws this and the validation
-- study quantifies what stopping early would cost.
with boundary as (
    select min(assigned_at) as experiment_start from {{ ref('stg_assignment') }}
),

daily as (
    select
        a.variant,
        s.session_date,
        count(*)                                  as sessions,
        count(distinct s.user_id)                 as users,
        sum(s.purchases)                          as purchases,
        sum(s.revenue)                            as revenue
    from {{ ref('int_sessions') }} s
    inner join {{ ref('stg_assignment') }} a using (user_id)
    cross join boundary b
    -- Experiment-period sessions only. The corpus also holds a pre-experiment window, which exists
    -- to give CUPED a covariate; including it here mixes in sessions from before the treatment
    -- existed, which dilutes the effect and can reverse its sign. A dashboard built on that would
    -- disagree with the readout while both looked plausible, so the two are tied together by
    -- dbt/tests/assert_daily_matches_user_metrics.sql.
    where s.session_start_ts >= b.experiment_start
    group by a.variant, s.session_date
),

cumulative as (
    select
        variant,
        session_date,
        sessions,
        users,
        purchases,
        revenue,
        sum(sessions)  over w as cumulative_sessions,
        sum(purchases) over w as cumulative_purchases,
        sum(revenue)   over w as cumulative_revenue
    from daily
    window w as (partition by variant order by session_date
                 rows between unbounded preceding and current row)
)

select
    *,
    cumulative_purchases::double / nullif(cumulative_sessions, 0) as cumulative_purchase_rate,
    cumulative_revenue / nullif(cumulative_sessions, 0)           as cumulative_revenue_per_session
from cumulative
order by variant, session_date
