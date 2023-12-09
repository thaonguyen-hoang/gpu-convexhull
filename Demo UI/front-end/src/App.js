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
    const [markedDone, setMarkedDone] = useState(false);

    const [points, setPoints] = useState(featureCollection([]));
    const [centroid, setCentroid] = useState(featureCollection([]));

    var count = 0;
    var hull;


    useEffect(() => {
        if (map.current) return;
        map.current = new mapboxgl.Map({
            container: mapContainer.current,
            style: 'mapbox://styles/mapbox/streets-v12',
            center: [105.750556856, 20.989612055],
            zoom: 16,
        });
        map.current.on('load', async () => {
        });


        if (!markedDone) {
            mapClick.current = handleMapClick;
            map.current.on('click', mapClick.current);
        }

        const geocoder = new MapboxGeocoder({
            accessToken: mapboxgl.accessToken,
            mapboxgl: mapboxgl,
            placeholder: 'Search...',
            bbox: [105.44, 20.53, 106.02, 21.23],
            countries: 'VN',
            language: 'vi-VN',
            enableHighAccuracy: true,
        });

        map.current.addControl(geocoder, 'top-right');

        const navigationControl = new mapboxgl.NavigationControl();
        map.current.addControl(navigationControl);

        const geolocateControl = new mapboxgl.GeolocateControl({
            positionOptions: {
                enableHighAccuracy: true,
            },
            trackUserLocation: true,
        });
        map.current.addControl(geolocateControl);

    }, []);

    // choose position by click
    function handleMapClick(e) {

        map.current.addLayer({
            id: `dropoff-${count}`,
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

        map.current.addLayer({
            id: `dropoff-${count}-symbol`,
            type: 'symbol',
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
            layout: {
                'icon-size': 0.5,
                'text-field': `${e.lngLat.lng}, ${e.lngLat.lat}`,
                'text-font': ['Arial Unicode MS Bold'],
                'text-offset': [0, 0.9],
                'text-anchor': 'top',
                'text-size' : 10
            },

        });

        count = count + 1;
        points.features.push({
            type: 'Feature',
            geometry: {
                type: 'Point',
                coordinates: [e.lngLat.lng, e.lngLat.lat]
            }
        });


    }

    // return convex hull
    function getConvexHull() {
        hull = turf.convex(points);
        // console.log(hull)
        // console.log(hull.geometry.coordinates[0][1])
        // console.log(hull.geometry.coordinates[0].length)


        const hullCoor = []
        for (let i=0; i<hull.geometry.coordinates[0].length; i++) {
            hullCoor.push([hull.geometry.coordinates[0][i]])
        }
        // console.log(hullCoor)

        map.current.addLayer({
            id: `hull`,
            type: 'fill',
            source: {
                data: {
                    type: 'Feature',
                    geometry: {
                        type: 'Polygon',
                        coordinates: hullCoor
                    },
                },
                type: 'geojson'
            },

            paint: {
                'fill-color': '#333',
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
                'circle-color': '#333',
            },
        });
        // console.log(centroid)


    }


    // reset
    function handleReset() {
        for (let i = 0; i < points.features.length; i++) {
            map.current.removeLayer(`dropoff-${i}`);
            map.current.removeSource(`dropoff-${i}`);
            map.current.removeLayer(`dropoff-${i}-symbol`);
            map.current.removeSource(`dropoff-${i}-symbol`);
        }
        points.features = [];
        centroid.features = []
        setMarkedDone(false);

        if (map.current.getLayer('centroid')) {
            map.current.removeLayer('centroid');
            map.current.removeSource('centroid');
        }
        if (map.current.getLayer('hull')) {
            map.current.removeLayer('hull');
            map.current.removeSource('hull');
        }
        if (mapClick.current) {
            map.current.off('click', mapClick.current);
        }
        mapClick.current = handleMapClick;
        map.current.on('click', mapClick.current);

        count = 0;
    }


    function handleConfirm() {
        setMarkedDone(true);
        if (mapClick.current) {
            map.current.off('click', mapClick.current);
        }
        getConvexHull()
        getCentroid()
    }

    return (
        <div>
            <div className="sidebar">

                {!markedDone && (
                    <button onClick={handleConfirm}>Confirm</button>
                )}
                {markedDone && (
                    <div>
                        <h2> Wi-fi Location </h2>

                        <h2>Input Points:</h2>
                        <ol>
                            {points.features.map((geo, index) => (
                                <li key={index}>{geo.geometry.coordinates[0]}, {geo.geometry.coordinates[1]}</li>
                            ))}
                        </ol>

                        <h2>Loaction of the wi-fi station: </h2>
                        <h3>{centroid.features["0"].geometry.coordinates[0]} , {centroid.features["0"].geometry.coordinates[0] } </h3>


                        <button onClick={handleReset}>Reset</button>
                    </div>
                )}
            </div>
            <div ref={mapContainer} className="map-container" />
        </div>
    );
}

export default App
