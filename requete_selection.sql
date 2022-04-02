DECLARE @start_date DATE
DECLARE @end_date DATE
SET @start_date = '2012-03-22'
SET @end_date = DATEADD(m , 12 , @start_date)

select
    p.Id,
    p.CreationDate,
    p.Title,
    p.Body,
    p.Tags,
    p.ViewCount,
    p.CommentCount,
    p.AnswerCount,
    p.Score,      
    sum(case when VoteTypeId = 2 then 1 else 0 end) as [up] ,
    sum(case when VoteTypeId = 3 then 1 else 0 end) as [down]
                                         
from Votes v join Posts p on v.PostId = p.Id
LEFT JOIN PostTypes as t ON p.PostTypeId = t.id
group by 
    p.Id, 
    p.CreationDate,
    p.Title,
    p.Body,
    p.Tags,
    p.ViewCount,
    p.CommentCount,
    p.AnswerCount,
    p.Score,
    v.VoteTypeId,
    t.Name 
HAVING v.VoteTypeId in (2,3)
AND p.CreationDate between @start_date and @end_date
AND t.Name = 'Question'
AND p.ViewCount > 20
AND p.CommentCount > 5
AND p.AnswerCount > 1
AND p.Score > 5
AND len(p.Tags) > 0
ORDER BY [up] desc
