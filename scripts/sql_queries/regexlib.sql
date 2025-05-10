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
;


-- get all regexes
select count(r.pattern)
from regexes r
where
	r.pattern not in ('', ' ')
	and r.pattern is not null
;


-- get completely similar regexes
select r.pattern, count(*) from regexes r
group by r.pattern
order by 2 desc;


-- get similar regexes with same construction
select r.pattern, count(*) from (
	select lower(pattern) as pattern from regexes
) as r
where r.pattern like '%\w%'
group by r.pattern
order by 2 desc;


-- get similar regexes with same construction
-- filter by percentage of the construction from the string
with vars as (select '.*' as pattern)
select
	r.pattern,
	r.pattern_percentage,
	count(*) as count
from (
	select
		lower(regexes.pattern) as pattern,
		round(
			cast(length(vars.pattern) as real)/length(regexes.pattern),
			2
		) as pattern_percentage
	from regexes, vars
) as r, vars
where
	r.pattern like format("%s%s%s", '%', vars.pattern, '%')
	and r.pattern_percentage >= 0.5
group by r.pattern
order by 2 desc, 3 desc
;
