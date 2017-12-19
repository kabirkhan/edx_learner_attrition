from datetime import datetime
import luigi
import pandas as pd 
from orchestra_pipeline.query_data import RawUserActivity, CourseDatesQueryTask
from orchestra_pipeline.augment_features import AddNegativeDataPoints
from orchestra_pipeline.util import course_week


class Pipeline(luigi.Task):
    course_id = luigi.Parameter()
    def requires(self):
        return CourseDatesQueryTask(self.course_id)

    def complete(self):
        return False
    
    def run(self):
        course_start_date, _ = self.get_course_dates()
        current_course_week = course_week(datetime.utcnow(), course_start_date)
        
        yield AddNegativeDataPoints(course_id=self.course_id, 
                                    course_week=current_course_week,
                                    course_start_date=course_start_date)

    def get_course_dates(self):
        with self.input().open() as course_dates_file:
            course_dates = pd.read_csv(course_dates_file)
            def get_datetime_col(col_name):
                """
                Get column as a datetime object
                """
                return datetime.strptime(course_dates[col_name][0], '%Y-%m-%d')

            course_start_date = get_datetime_col('CourseRunStartDate')
            course_end_date = get_datetime_col('CourseRunEndDate')
            return (course_start_date, course_end_date)


if __name__ == "__main__":
    luigi.run()
