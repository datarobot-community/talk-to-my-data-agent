# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Hashable

import pandas as pd
from pandas.testing import assert_frame_equal
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class AnalystDataset(BaseModel):
    name: str
    data: pd.DataFrame = Field(default_factory=pd.DataFrame)  # Removed exclude=True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={pd.DataFrame: lambda df: df.to_dict(orient="records")},
    )

    @field_validator("data", mode="before")
    @classmethod
    def validate_dataframe(cls, v: Any) -> pd.DataFrame:
        if isinstance(v, pd.DataFrame):
            return v
        elif isinstance(v, list):
            try:
                return pd.DataFrame.from_records(v)
            except Exception as e:
                raise ValueError("Invalid data format") from e
        else:
            raise ValueError("data has to be either list of records or pd.DataFrame")

    def to_df(self) -> pd.DataFrame:
        return self.data

    @property
    def columns(self) -> list[str]:
        return self.data.columns.tolist()

    @computed_field
    def data_records(self) -> list[dict[Hashable, Any]]:
        return self.data.to_dict(orient="records")


# Example usage:
if __name__ == "__main__":
    # Create a DataFrame.
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    data = {"df": df}

    # Initialize the model.
    model = AnalystDataset(name="test", data=data["df"])

    # Serialize the model to JSON.
    serialized = model.model_dump_json()
    print(serialized)

    # Deserialize back into an AnalystDataset instance.
    deserialized = AnalystDataset.model_validate_json(serialized)

    # Verify that the deserialized DataFrame matches the original.
    assert_frame_equal(deserialized.to_df(), data["df"])
    print("DataFrame round-trip successful!")
