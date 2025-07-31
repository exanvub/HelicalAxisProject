# HelicalAxisProject

This repository is dedicated to the collaborative development of the Helical Axis Method for describing motion in human joints. The project aims to explore different approaches and techniques to solve the helical axis problem.

Please note that this project is currently a work in progress and is by no means finished. Contributions from the community are highly encouraged and welcomed.

# Finite Helical Axis (FHA) Analysis Toolkit

The toolkit includes vectorized methods for computing FHA parameters from time-series transformation matrices, with multiple options for time-based and angle-based FHA segmentation. Interactive 3D visualization is included using **Plotly** and **Dash**.

---

## Features

* Efficient **vectorized FHA calculation** between pairs of homogeneous transformation matrices.
* Multiple analysis modes:

  * **All FHA**: Instantaneous FHA between all consecutive frames.
  * **Incremental Time**: FHA computed over fixed time steps.
  * **Step Angle**: FHA computed after crossing a cumulative angle threshold.
  * **Incremental Angle**: FHA computed after incrementally increasing rotation.
* **Global frame transformation** of axes and origins.
* **Plane intersection calculation** for spatial referencing.
* **Interactive 3D visualization** of helical axes and motion using Plotly/Dash.
* Includes tools for:

  * Axis direction computation
  * Rotation angle extraction
  * Axis of average motion estimation

---

## Input Requirements

To use the FHA analysis functions, provide the following:

* **Transformation Matrices**: Two synchronized lists or arrays of homogeneous 4x4 matrices (`T1`, `T2`) representing the poses of a body segment over time.
* **Time Vector**: A time series (`t`) aligned with the transformations.
* **Configuration Parameters**:

These can be adjusted in the config.py file
  * `method_type`: One of `'all_FHA'`, `'incremental_time'`, `'step_angle'`, or `'incremental_angle'`.
  * `step`: Step size in seconds or degrees, depending on the method.
  * `cut1`, `cut2`: Optional index range to cut the data.

---

## Output

Each FHA calculation method returns:

* `hax`: FHA direction vectors
* `ang`: Rotation angles around each axis
* `svec`: Points on the FHA
* `d`: Translation magnitudes along the axis
* `translation_1_list`, `translation_2_list`: Associated translations for each segment
* `time_diff`, `time_incr`, `ang_incr`: Time and angle intervals
* `ind_incr`, `ind_step`: Indices used for incremental and step methods
* `t`: Corresponding time vector

---

## Visualization

An interactive 3D Plotly dashboard displays:

* Instantaneous or average FHA vectors in global coordinates
* Points of origin and translation
* A reference plane orthogonal to the average axis
* Euler angles (for relevant methods)
* Optional data overlays (if using Polhemus tracking or custom datasets)

The visualization helps identify dominant motion patterns and axis stability.

---

## Contact

For questions or contributions, please contact **nicolas.van.vlasselaer@vub.be** or **matteo.iurato@edu.unige.it**

---

This code is distributed for academic and research purposes only. Please cite appropriately if used in publications.