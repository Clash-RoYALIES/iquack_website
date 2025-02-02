N = 16           # Number of facilities/locations
T = 4           # Number of time steps (e.g., months)
alpha = 1.0      # Movement cost scaling factor

d = [
    [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    [10, 0, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38],
    [11, 25, 0, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
    [12, 26, 39, 0, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
    [13, 27, 40, 52, 0, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74],
    [14, 28, 41, 53, 64, 0, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84],
    [15, 29, 42, 54, 65, 75, 0, 85, 86, 87, 88, 89, 90, 91, 92, 93],
    [16, 30, 43, 55, 66, 76, 85, 0, 94, 95, 96, 97, 98, 99, 100, 101],
    [17, 31, 44, 56, 67, 77, 86, 94, 0, 102, 103, 104, 105, 106, 107, 108],
    [18, 32, 45, 57, 68, 78, 87, 95, 102, 0, 109, 110, 111, 112, 113, 114],
    [19, 33, 46, 58, 69, 79, 88, 96, 103, 109, 0, 115, 116, 117, 118, 119],
    [20, 34, 47, 59, 70, 80, 89, 97, 104, 110, 115, 0, 120, 121, 122, 123],
    [21, 35, 48, 60, 71, 81, 90, 98, 105, 111, 116, 120, 0, 124, 125, 126],
    [22, 36, 49, 61, 72, 82, 91, 99, 106, 112, 117, 121, 124, 0, 127, 128],
    [23, 37, 50, 62, 73, 83, 92, 100, 107, 113, 118, 122, 125, 127, 0, 129],
    [24, 38, 51, 63, 74, 84, 93, 101, 108, 114, 119, 123, 126, 128, 129, 0]
]

flow = {
0:[
[0, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2 ],
[1, 0, 1, 0, 0, 2, 1, 1, 2, 0, 1, 1, 1, 1, 1, 2],
[1, 1, 0, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2],
[1, 0, 1, 0, 1, 1, 1, 1, 2, 0, 2, 2, 2, 2, 2, 3],
[1, 0, 2, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2],
[2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3],
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3],
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3],
[2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3],
[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 3],
[1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0]
],
    1:[
[0, 2, 2, 2, 2, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 2],
[2, 0, 2, 2, 2, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 2],
[2, 2, 0, 2, 2, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 2],
[2, 2, 2, 0, 2, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 2],
[2, 2, 2, 2, 0, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 2],
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2],
[1, 1, 1, 1, 1, 0.5, 0.25, 0.25, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 0.5, 0.25, 0.25, 0.5, 0.5, 1, 1, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 0.5, 0.25, 0.25, 0.5, 0.5, 1, 1, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 0.5, 0.25, 0.25, 0.5, 0.5, 1, 1, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 0.5, 0.25, 0.25, 0.5, 0.5, 1, 1, 0, 0, 0, 0, 0, 0],
[1.5, 1.5, 1.5, 1.5, 1.5, 1, 0.5, 0.5, 1, 1, 0, 0, 0, 0, 0, 0]
],

    2:[
[0, 2, 2, 2, 2, 4, 2, 2, 4, 3, 2, 2, 2, 2, 2, 3 ],
[2, 0, 2, 2, 2, 4, 2, 2, 4, 3, 2, 2, 2, 2, 2, 3],
[2, 2, 0, 2, 2, 4, 2, 2, 4, 3, 2, 2, 2, 2, 2, 3],
[2, 2, 2, 0, 2, 4, 2, 2, 4, 3, 2, 2, 2, 2, 2, 3],
[2, 2, 2, 2, 0, 4, 2, 2, 4, 3, 2, 2, 2, 2, 2, 3],
[3, 3, 3, 3, 3, 0, 3, 2, 4, 2, 1, 1, 1, 1, 1, 2],
[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2],
[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2],
[3, 3, 3, 3, 3, 4, 3, 2, 0, 2, 1, 1, 1, 1, 1, 2],
[3, 3, 3, 3, 3, 3, 2, 2, 3, 0, 1, 1, 1, 1, 1, 2],
[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0]
],

    3:[
[0, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
[2, 0, 2, 1, 0, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3],
[0, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
[1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4],
[2, 0, 2, 1, 0, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3],
[2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4],
[2, 1, 2, 2, 1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4],
[2, 1, 2, 2, 1, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4],
[2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4],
[2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4],
[2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
[2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0],
[3, 3, 3, 4, 3, 4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0]
]

}