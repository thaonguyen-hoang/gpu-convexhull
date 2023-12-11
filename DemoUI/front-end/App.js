import React, {useEffect, useRef, useState} from 'react';

import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import '@mapbox/mapbox-gl-geocoder/dist/mapbox-gl-geocoder.css';
import MapboxGeocoder from '@mapbox/mapbox-gl-geocoder';

import * as turf from '@turf/turf';
import {featureCollection} from '@turf/turf';

import './index.css';

const MAPBOX_API_KEY = "pk.eyJ1IjoidGhhbmhoYWlpMDMiLCJhIjoiY2xwZ3R0dTJpMDFmODJxbGZpMTB0bG93dCJ9.zPguGUAukU-bfVTDnW-NlQ"
mapboxgl.accessToken = MAPBOX_API_KEY;


function App() {
    const map = useRef(null);
    const mapContainer = useRef(null);
    const mapClick = useRef(null);
    const [Click, setClick] = useState(false);

    const [points, setPoints] = useState(featureCollection([]));
    const [centroid, setCentroid] = useState(featureCollection([]));

    var numPoint = 0;
    var hull;


    // set up map
    useEffect(() => {
        if (map.current) return;
        map.current = new mapboxgl.Map({
            container: mapContainer.current,
            style: 'mapbox://styles/mapbox/streets-v12',
            center: [105.750556856, 20.989612055],
            zoom: 16,
        });

        if (!Click) {
            mapClick.current = controlClick;
            map.current.on('click', mapClick.current);
        }


        const mapBoxgeocoder = new MapboxGeocoder({
            accessToken: mapboxgl.accessToken,
            mapboxgl: mapboxgl,
            placeholder: 'Search...'
        });

        map.current.addControl(mapBoxgeocoder, 'top-right');

        const navigation = new mapboxgl.NavigationControl();
        map.current.addControl(navigation);

        const geolocation = new mapboxgl.GeolocateControl({
            trackUserLocation: true,
        });
        map.current.addControl(geolocation);


    });

    // choose position by click
    function controlClick(e) {

        map.current.addLayer({
            id: `point-${numPoint}`,
            type: 'circle',
            source: {
                data: {
                    type: 'FeatureCollection',
                    features: [
                        {
                            type: 'Feature',
                            geometry: {
                                type: 'Point',
                                coordinates: [e.lngLat.lng, e.lngLat.lat]
                            }
                        }
                    ]
                },
                type: 'geojson'
            },

            paint: {
                'circle-radius': 7,
                'circle-color': '#3081D0',
            },
        });

        points.features.push({
            type: 'Feature',
            geometry: {
                type: 'Point',
                coordinates: [e.lngLat.lng, e.lngLat.lat]
            }
        });

        numPoint += 1;

    }

    // return convex hull
    function getConvexHull() {
        hull = turf.convex(points);
        console.log(hull)
        // console.log(hull.geometry.coordinates[0][1])
        // console.log(hull.geometry.coordinates[0].length)

        const hullCoor = []
        for (let i=0; i<hull.geometry.coordinates[0].length; i++) {
            const coor = [hull.geometry.coordinates[0][i]]
            hullCoor.push(coor)
        }
        console.log(hullCoor)

        map.current.addLayer({
            id: `hull`,
            type: 'circle',
            source: {
                data: {
                    type: 'Feature',
                    geometry: {
                        type: 'Polygon',
                        coordinates: [hull.geometry.coordinates[0]]
                    }
                },
                type: 'geojson'
            },

            paint: {
                'circle-radius': 10,
                'circle-color': '#7E30E1',
            },
        });


    }

    // return centroid
    function getCentroid() {
        const cen = turf.centroid(hull)
        // console.log(cen)

        centroid.features.push({
            type: 'Feature',
            geometry: {
                type: 'Point',
                coordinates: [cen.geometry.coordinates[0], cen.geometry.coordinates[1]]
            }
        });

        // draw centroid
        map.current.addLayer({
            id: `centroid`,
            type: 'circle',
            source: {
                data: {
                    type: 'Feature',
                    geometry: {
                        type: 'Point',
                        coordinates: centroid.features["0"].geometry.coordinates
                    }
                },
                type: 'geojson'
            },

            paint: {
                'circle-radius': 10,
                'circle-color': '#EF4040',
            },
        });
        // console.log(centroid)


    }

    function Confirm() {
        if (points.features.length > 2) {
            setClick(true);
            if (mapClick.current) {
                map.current.off('click', mapClick.current);
            }
            getConvexHull()
            getCentroid()
        }

    }


    // reset
    function Reset() {
        for (let i = 0; i < points.features.length; i++) {
            map.current.removeLayer(`point-${i}`);
            map.current.removeSource(`point-${i}`);
        }
        if (map.current.getLayer('centroid')) {
            map.current.removeLayer('centroid');
            map.current.removeSource('centroid');
            map.current.removeLayer('hull');
            map.current.removeSource('hull');
        }

        points.features = [];
        centroid.features = []
        numPoint = 0;

        mapClick.current = controlClick;
        map.current.on('click', mapClick.current);
        setClick(false);

    }



    return (
        <div className='menu'>
            <div className="sidebar">
                <h1 style={{fontSize : 50, textAlign : "center" }}>Wi-fi Location </h1>
                <div className="image-logo">
                    <img src = "static/wifi.png" alt={"logo"} width={200} height={200}  />
                </div>
                <p style={{textAlign:"center", fontSize: 20}}>Click on the map to choose the location</p>

                {!Click && (
                    <button onClick={Confirm}>Confirm</button>
                )}

                {Click && (
                    <div>
                        <h2>Input Points:</h2>
                            {points.features.map((point, index) => (
                                <li key={index}>{point.geometry.coordinates[0]}, {point.geometry.coordinates[1]}</li>
                            ))}

                        <h2>Location of the wi-fi station: </h2>
                        <h3 style={{background: "lightskyblue", alignContent:"center"}}>{centroid.features["0"].geometry.coordinates[0]} , {centroid.features["0"].geometry.coordinates[0] } </h3>
                        <button onClick={Reset}>Reset</button>
                    </div>
                )}
            </div>

            <div ref={mapContainer} className="map-container" />
        </div>
    );
}

export default App
