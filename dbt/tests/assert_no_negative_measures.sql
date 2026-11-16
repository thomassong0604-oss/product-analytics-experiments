-- Counts and revenue cannot be negative. A negative here means a join fanned out and a subtraction
-- went the wrong way, which is the kind of error that produces plausible-looking totals.
select user_id, sessions, purchases, revenue
from {{ ref('fct_user_metrics') }}
where sessions < 0 or purchases < 0 or revenue < 0
