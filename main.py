from behavior_analysis import BoutData
import pandas as pd
from pathlib import Path
#
# data_path = Path(r"D:\DATA\Mearns2020")
#
# if __name__ == "__main__":
#     exemplar_md = pd.read_csv(data_path.joinpath("exemplars.csv"), dtype={"ID": str, "video_code": str})
#     exemplar_md = exemplar_md[exemplar_md["clean"]]
#     bout_data = BoutData.from_metadata(exemplar_md.iloc[:100], data_path.joinpath("kinematics"),
#                                        tail_only=True)
from ethomap import DynamicTimeWarping

help(DynamicTimeWarping)
