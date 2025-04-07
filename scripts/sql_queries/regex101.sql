-- see few rows
select * from regexes r;


-- see table structure
PRAGMA table_info(regexes);


-- see regex number
select count(*) from regexes r;


-- get distribution for flavors
select r.dialect, count(*) as regexes_number from regexes r
group by r.dialect;


-- get average, maximum and minimum length of regexes by flavors
with regexes_length as (
	select
		r.dialect,
		length(r.regex) as reg_length
	from regexes r
	where reg_length > 0
)
select
	rl.dialect,
	max(rl.reg_length) as max_length,
	min(rl.reg_length) as min_length,
	avg(rl.reg_length) as avg_length
from regexes_length rl
group by rl.dialect
;


-- get maximum regexes
select
	r.regex,
	r.test_string,
	r.title,
	r.description,
	r.dialect,
	length(r.regex) as regexes_length
from regexes r
order by length(r.regex) desc;


-- get all regexes with "SQL" into title for PCRE flavor
with pcre_regexes as (
	select
		r.regex,
		r.test_string,
		lower(r.title) as title,
		lower(r.description) as description
	from regexes r
	where r.dialect = 'pcre'
)
select * from pcre_regexes pr
where pr.title like '%sql%'
;