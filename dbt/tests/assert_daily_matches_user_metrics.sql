-- The daily curve and the user-level fact table must describe the same experiment.
--
-- They are built by different paths — one aggregates sessions by day, the other pivots a per-user
-- period table — so nothing but this test stops them drifting apart. They did drift once: the daily
-- model was including pre-experiment sessions, which reversed the sign of the headline metric while
-- every number still looked reasonable on its own.
with daily as (
    select
        variant,
        max(cumulative_purchases) as purchases,
        max(cumulative_sessions)  as sessions
    from {{ ref('mart_experiment_daily') }}
    group by variant
),

per_user as (
    select variant, sum(purchases) as purchases, sum(sessions) as sessions
    from {{ ref('fct_user_metrics') }}
    group by variant
)

select
    d.variant,
    d.purchases as daily_purchases,
    u.purchases as user_purchases,
    d.sessions  as daily_sessions,
    u.sessions  as user_sessions
from daily d
inner join per_user u using (variant)
where d.purchases <> u.purchases or d.sessions <> u.sessions
