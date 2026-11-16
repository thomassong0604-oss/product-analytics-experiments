-- RFM segmentation: recency, frequency and monetary value, each split into quintiles.
--
-- ntile over the whole population rather than fixed thresholds, so the segments stay meaningful as
-- the business grows instead of drifting until everyone lands in one bucket. Recency is reversed —
-- a recent visit is a *high* score — which is the usual source of an inverted RFM table.
with per_user as (
    select
        u.user_id,
        u.country,
        u.channel,
        max(s.session_start_ts)                                  as last_seen,
        date_diff('day', max(s.session_start_ts),
                  (select max(session_start_ts) from main_intermediate.int_sessions))
                                                                 as days_since_last_session,
        count(*)                                                 as sessions,
        sum(s.purchases)                                         as purchases,
        sum(s.revenue)                                           as revenue
    from main_marts.fct_user_metrics u
    inner join main_intermediate.int_sessions s using (user_id)
    group by u.user_id, u.country, u.channel
),

scored as (
    select
        *,
        6 - ntile(5) over (order by days_since_last_session)     as recency_score,
        ntile(5) over (order by sessions)                        as frequency_score,
        ntile(5) over (order by revenue)                         as monetary_score
    from per_user
)

-- Grouped by segment, not by the 125 score combinations: a table nobody can read is not a
-- segmentation. The mean scores are kept so each segment's shape stays visible.
select
    case
        when recency_score >= 4 and frequency_score >= 4 and monetary_score >= 4 then 'champions'
        when recency_score >= 4 and frequency_score <= 2                         then 'new or promising'
        when recency_score <= 2 and monetary_score >= 4                          then 'at risk, high value'
        when recency_score <= 2 and frequency_score <= 2                         then 'lapsed'
        else 'everyone else'
    end                                              as segment,
    count(*)                                         as users,
    round(avg(recency_score), 2)                     as mean_recency_score,
    round(avg(frequency_score), 2)                   as mean_frequency_score,
    round(avg(monetary_score), 2)                    as mean_monetary_score,
    round(avg(days_since_last_session), 1)           as mean_days_since_last_session,
    round(avg(sessions), 2)                          as mean_sessions,
    round(avg(revenue), 2)                           as mean_revenue,
    round(sum(revenue), 2)                           as total_revenue,
    round(100 * sum(revenue) / sum(sum(revenue)) over (), 1) as share_of_revenue_pct
from scored
group by segment
order by total_revenue desc
