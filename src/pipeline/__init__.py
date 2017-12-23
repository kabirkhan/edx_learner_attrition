from datetime import datetime
import luigi
import pandas as pd
from common import ADLFlagTask
from pipeline.query_data import RawUserActivity, CourseDatesQueryTask
from pipeline.augment_features import AddNegativeDataPoints
from pipeline.util import course_week


class Pipeline(ADLFlagTask):
    """
    Run a full data transformation for one course.
    Complete if a `_SUCCESS` flag file exists in the
    the course folder's current week.
    e.g. /data/{course_id}/week_6/_SUCCESS
    """
    course_id = luigi.Parameter()
    course_start_date = luigi.DateParameter()
    current_course_week = luigi.DateParameter()
    
    def requires(self):
        return AddNegativeDataPoints(course_id=self.course_id, 
                                     course_week=self.current_course_week,
                                     course_start_date=self.course_start_date)


if __name__ == "__main__":
    luigi.run()
