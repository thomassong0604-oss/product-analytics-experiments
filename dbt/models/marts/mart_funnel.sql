-- Step-by-step funnel conversion, by variant.
--
-- Two different rates, because they answer different questions and are routinely confused: the
-- step-to-step rate says how well one step performs, while the rate from the top says how much
-- total volume survives to here. A step can improve on the first and barely move the second.
with flags as (
    select
        a.variant,
        s.session_id,
        true                       as step_session_start,
        s.reached_product_view     as step_product_view,
        s.reached_add_to_cart      as step_add_to_cart,
        s.reached_checkout_start   as step_checkout_start,
        s.reached_purchase         as step_purchase
    from {{ ref('int_sessions') }} s
    inner join {{ ref('stg_assignment') }} a using (user_id)
),

counted as (
    select variant,
           count(*)                                       as sessions,
           count(*) filter (where step_product_view)       as product_view,
           count(*) filter (where step_add_to_cart)        as add_to_cart,
           count(*) filter (where step_checkout_start)     as checkout_start,
           count(*) filter (where step_purchase)           as purchase
    from flags
    group by variant
),

unpivoted as (
    select variant, 0 as step_number, 'session_start'  as step, sessions       as reached, sessions as top from counted
    union all
    select variant, 1, 'product_view',   product_view,   sessions from counted
    union all
    select variant, 2, 'add_to_cart',    add_to_cart,    sessions from counted
    union all
    select variant, 3, 'checkout_start', checkout_start, sessions from counted
    union all
    select variant, 4, 'purchase',       purchase,       sessions from counted
)

select
    variant,
    step_number,
    step,
    reached,
    -- lag over the ordered steps gives the previous step's volume without a self-join.
    lag(reached) over (partition by variant order by step_number) as reached_previous_step,
    reached::double / nullif(lag(reached) over (partition by variant order by step_number), 0)
        as step_conversion,
    reached::double / nullif(top, 0) as conversion_from_top,
    1 - reached::double / nullif(lag(reached) over (partition by variant order by step_number), 0)
        as step_dropoff
from unpivoted
order by variant, step_number
