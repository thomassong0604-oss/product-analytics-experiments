-- Where the funnel leaks differently by device and channel.
--
-- An aggregate funnel hides the only thing worth acting on: a step that is fine overall and broken
-- on one surface. Conditional aggregation keeps it to a single pass, and the segments are kept
-- side by side so the comparison is a row scan rather than four queries.
select
    u.device,
    u.channel,
    count(*)                                                         as sessions,
    count(distinct s.user_id)                                        as users,
    round(avg(s.reached_add_to_cart::int), 4)                        as cart_rate,
    round(avg(s.reached_checkout_start::int), 4)                     as checkout_rate,
    round(avg(s.reached_purchase::int), 4)                           as purchase_rate,
    -- Of those who started checkout, how many finished. The step most worth fixing, because the
    -- user has already declared intent.
    round(count(*) filter (where s.reached_purchase)::double
          / nullif(count(*) filter (where s.reached_checkout_start), 0), 4) as checkout_completion,
    round(sum(s.revenue), 2)                                         as revenue,
    round(sum(s.revenue) / count(*), 2)                              as revenue_per_session
from main_intermediate.int_sessions s
inner join main_marts.fct_user_metrics u using (user_id)
group by u.device, u.channel
having count(*) >= 100
order by checkout_completion
