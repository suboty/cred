-- see few rows
select * from regexes r;


-- see table structure
PRAGMA table_info(regexes);


-- see regex number
select count(*) from regexes r;


-- get distribution for ratings
select r.rating, count(*) as regexes_number from regexes r
group by r.rating;


-- get average, maximum and minimum length of regexes by rating
with regexes_length as (
	select
		r.rating,
		length(r.pattern) as reg_length
	from regexes r
	where reg_length > 0
)
select
	rl.rating,
	max(rl.reg_length) as max_length,
	min(rl.reg_length) as min_length,
	avg(rl.reg_length) as avg_length
from regexes_length rl
group by rl.rating
;


-- get all regexes with "mail address" string
select
	r.pattern,
	r.rating,
	r.title,
	r.description
from regexes r
where
	lower(r.title) like '%mail_address%'
	or lower(r.description) like '%mail_address%'