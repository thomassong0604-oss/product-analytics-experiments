-- Weekly cohort retention: of the users who signed up in week W, how many were active N weeks later.
--
-- The cohort size is taken over the whole cohort with a window function, so a week in which nobody
-- returned still divides by the right denominator. Computing it from the active rows instead — the
-- common mistake — silently rebases every row and makes retention look flat.
with activity as (
    select distinct
        u.user_id,
        u.cohort_week,
        date_trunc('week', s.session_start_ts) as active_week
    from {{ ref('stg_users') }} u
    inner join {{ ref('int_sessions') }} s using (user_id)
),

indexed as (
    select
        cohort_week,
        user_id,
        active_week,
        date_diff('week', cohort_week, active_week) as weeks_since_signup
    from activity
    where active_week >= cohort_week
),

cohort_sizes as (
    select cohort_week, count(*) as cohort_users
    from {{ ref('stg_users') }}
    group by cohort_week
)

select
    i.cohort_week,
    i.weeks_since_signup,
    c.cohort_users,
    count(distinct i.user_id)                                   as active_users,
    count(distinct i.user_id)::double / nullif(c.cohort_users, 0) as retention
from indexed i
inner join cohort_sizes c using (cohort_week)
group by i.cohort_week, i.weeks_since_signup, c.cohort_users
order by i.cohort_week, i.weeks_since_signup
