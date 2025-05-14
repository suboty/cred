-- get tables names
select ss.name from sqlite_schema ss
where ss."type" = 'table';

-- create results view
-- for example%
-- regex table: regexes_1111
-- result talbe: results_1111
-- "1111" is timestamp in real tables

create view result_view as select
	r.regex_type,
	r2.regex as regex1,
	r3.regex as regex2,
	r.metric_name,
	r.metric_value
from results_1111 r
join regexes_1111 r2 on r.regex1 = r2.id
join regexes_1111 r3 on r.regex2 = r3.id
;

-- calculates metrics
select
	f.regex_type,
	f.metric_name,
	case
		when
			(f.regex_type = 'str' and f.metric_name = 'levenshtein')
			or (f.regex_type = 'str' and f.metric_name = 'hamming')
		then sum(abs(0-f.metric_value))
		when
			(f.regex_type = 'str' and f.metric_name = 'jaro')
			or (f.regex_type = 'str' and f.metric_name = 'jaro_winkler')
		then sum(abs(1-f.metric_value))
		when
			f.regex_type = 'ast'
		then sum(abs(0-f.metric_value))
	end as metric
from result_view f
group by 1, 2
order by 1, 2, 3
;