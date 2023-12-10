# Installation Guide For ”Wi-fi Location” Website

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
- Sau đó, vào file 'Map.js', thay token:
`const MAPBOX_API_KEY = "[your_token]"`
 #### Start the website
 Trong thư mục front_end, mở Terminal và chạy dòng lệnh :
 `npm start`
