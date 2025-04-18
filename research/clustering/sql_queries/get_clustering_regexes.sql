-- get clustering regexes
select regex, cluster_id from clustering_TIMEID c
join regexes_TIMEID r on c.regex_id = r.id
where c.experiment_id = BESTEXPID;