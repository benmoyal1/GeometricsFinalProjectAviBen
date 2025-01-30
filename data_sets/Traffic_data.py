import pandas as pd

file_path = r'C:\Users\Avi\OneDrive - huji.ac.il\שולחן העבודה\הכל\לימודים\לימודים 2025\גאומטריה של מידע\יישום\data_sets\us_congestion_2016_2022.csv'
save_file_path = r'C:\Users\Avi\OneDrive - huji.ac.il\שולחן העבודה\הכל\לימודים\לימודים 2025\גאומטריה של מידע\יישום\data_sets\Traffic_data.csv'



filtered_data = pd.read_csv(
    file_path, 
    usecols=["County", "State", "Start_Lat", "Start_Lng", "StartTime", "DelayFromTypicalTraffic(mins)", "ZipCode"]
)

filtered_data = filtered_data[
    (filtered_data["County"] == "Alameda") & 
    (filtered_data["State"] == "CA")
]

filtered_data = filtered_data[filtered_data["ZipCode"].str.contains("-")]##

zipcodes_with_delays = filtered_data[
    filtered_data["DelayFromTypicalTraffic(mins)"] > 8
]["ZipCode"].unique()

filtered_data = filtered_data[filtered_data["ZipCode"].isin(zipcodes_with_delays)]


filtered_data["StartTime"] = pd.to_datetime(filtered_data["StartTime"], utc=True)
filtered_data["Date"] = filtered_data["StartTime"].dt.date

all_dates = filtered_data["Date"]

filtered_data.rename(columns={"DelayFromTypicalTraffic(mins)": "DelayFromTypicalTrafficMins"}, inplace=True)



representative_coords = filtered_data.groupby("ZipCode").agg({
    "Start_Lat": "first",
    "Start_Lng": "first"
}).reset_index()

filtered_data = filtered_data.merge(representative_coords, on="ZipCode", suffixes=("", "_Representative"))


filtered_data = filtered_data.groupby(["ZipCode","Start_Lng_Representative","Start_Lat_Representative","Date"]).agg({
    "DelayFromTypicalTrafficMins": "sum",
}).reset_index()


filtered_data["Coordinates"] = filtered_data.apply(
    lambda row: f"({row['Start_Lat_Representative']}, {row['Start_Lng_Representative']})", axis=1
)

filtered_data.drop_duplicates(inplace=True)

unique_values_summary = filtered_data.nunique()

print("Unique values in each column:")
print(unique_values_summary)


missing_days = set(all_dates) - set(filtered_data["Date"].unique())
print(len(missing_days))

save_file_path = 'C:\Temp\Traffic_data.csv' # Permission issues
filtered_data.to_csv(save_file_path, index=False)


