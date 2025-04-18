-- get id of best experiment
select r.experiment_id from results_TIMEID r
join experiments_TIMEID e on r.experiment_id = e.id
order by r.metric_value desc
limit 1;