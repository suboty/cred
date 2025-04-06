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
select
	r.rating,
	count(*) as regexes_number,
	avg(length(r.pattern)) as avg_length,
	min(length(r.pattern)) as min_length,
	max(length(r.pattern)) as max_length
from regexes r
group by r.rating
;