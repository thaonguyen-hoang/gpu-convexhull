# Installation Guide

## Installation
#### NodeJS
Tải và cài đặt [NodeJS](https://nodejs.org/en) trên máy tính
#### Python Packages 
Cài đặt các thư viện python cần thiết cho backend 
- numba : `pip install numba` |  `conda install -c numba numba`
- cupy : ` pip install cupy` | `conda install -c conda-forge cupy`
- numpy : `pip install numpy`
- multiprocessing : `pip install multiprocess` 
- flask : `pip install Flask` | `conda install -c anaconda flask`
#### Node_modules
Trong thư mục front_end, sử dụng Terminal để cài đặt các modules cần thiết
`npm install`

## Run
#### Set up map
- Truy cập vào trang web [Mapbox](https://www.mapbox.com/), tạo tài khoản, đăng nhập để lấy API Access Token
- Sau đó, vào file 'App.js', thay token:
`const MAPBOX_API_KEY = "[your_token]"`
 #### Start the website
 Trong thư mục front_end, mở Terminal và chạy dòng lệnh :
 `npm start`

 # Usage Guide
 ## Giao diện chính
![main](https://github.com/duongthanhhaii/Nhom6_ConvexHull/assets/109170970/d4656f69-4a69-4873-8f6c-62945e3e7e4f)

## Chức năng chính
Chọn các điểm trên bản đồ làm tập input đầu vào bằng cách click trực tiếp trên biểu đồ
![point](https://github.com/duongthanhhaii/Nhom6_ConvexHull/assets/109170970/36c14b80-431f-402f-8060-999f060554f3)
Sau khi hoàn tất, bấm vào “Confirm”. Lúc này web trả về:
- Các đỉnh của convex hull : màu tím
- Tâm của convex hull, hay nơi đặt trạm phát wi-fi : màu đỏ
- ![hull](https://github.com/duongthanhhaii/Nhom6_ConvexHull/assets/109170970/64a16f03-df12-4fd5-a451-ffcea2fa75b4)

## Các chức năng khác
- Search địa điểm
![search](https://github.com/duongthanhhaii/Nhom6_ConvexHull/assets/109170970/f0d209c4-6a2d-4fd5-88c9-40e43649e91c)
- Thao tác với bản đồ
![other](https://github.com/duongthanhhaii/Nhom6_ConvexHull/assets/109170970/97331c33-7fa7-49ff-b0b0-ca24a975d357)









