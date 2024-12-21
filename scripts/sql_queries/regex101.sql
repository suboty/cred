-- see table structure
select * from regexes r;


-- get distribution for flavors
select r.dialect, count(*) as regexes_number from regexes r
group by r.dialect;


-- get average, maximum and minimum length of regexes by flavors
select
	r.dialect,
	count(*) as regexes_number,
	avg(length(r.regex)) as avg_length,
	min(length(r.regex)) as min_length,
	max(length(r.regex)) as max_length
from regexes r
group by r.dialect;


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