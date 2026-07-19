# Clarification on X,Y coordinate system and potential transformation to geographic coordinates

- 投稿者: Tahar HAMDAOUI
- 投稿日時: 2026-06-25 12:39:51.162000
- 投票数: 1
- コメント数: 2（取得数: 2）
- トピックID: `713987`
- 原文: [https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/713987](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/713987)

## 本文

<p>Hello,</p>
<p>Thank you for organizing this interesting competition. I have a question regarding the coordinate columns X (Easting) and Y (Northing).</p>
<p>Upon inspecting the data, I noticed the ranges of these coordinates appear inconsistent with standard projected coordinate systems typically used in Texas. For example, the Y values are around 1,000,000 ft, whereas in State Plane Texas Central (FIPS 4203, US Feet) the Y values near Austin are usually around 10,000,000 ft. Also, UTM coordinates in meters would have Y values in the range of several million.</p>
<p>This leads me to believe the coordinates are defined on a local grid (arbitrary origin) specific to the field, rather than a published geographic projection. Could you please confirm this?</p>
<p>Specifically:</p>
<p>Is this indeed a local grid, or is there a known projected coordinate system (e.g., a specific State Plane zone, UTM, or a custom projection) that these X,Y values correspond to?</p>
<p>If it is a local grid, can we assume the linear unit is international feet (or US survey feet), and the axes are aligned such that X represents local Easting and Y represents local Northing?</p>
<p>Is there any publicly available transformation (e.g., a set of parameters or a grid shift file) that would allow converting these local coordinates to a standard geographic system like WGS84 or UTM? Or should we treat them purely as relative coordinates and avoid any attempt to georeference them on global maps?</p>
<p>If treating them as relative only, do you have any recommendations for maintaining consistency when computing derived variables such as horizontal displacement or azimuth, considering that a local grid may have no true north alignment?</p>
<p>Any clarification would be greatly appreciated to avoid misinterpretation of the spatial component of the data. Thank you!</p>

## コメント

### コメント 1 — PC Jimmmy

- 投稿日時: 2026-06-26 12:57:55.627000
- 投票数: 0
- コメントID: `3481800`

<p>Don't think there is much chance you will be able to find the 773 wells - that would be a pretty leak as you would also be finding the test wells.  Its very common for kaggle to make changes to keep the data normalized in some fashion to keep folks with great google skills from finding more info than provided.</p>

### コメント 2 — Ayush Khaire

- 投稿日時: 2026-06-26 03:50:17.037000
- 投票数: 0
- コメントID: `3481453`

<p><a href="https://www.kaggle.com/taharhamdaoui">@taharhamdaoui</a> </p>
<p>Did you get anything ? Please ping me if you find something aaround it :)</p>
