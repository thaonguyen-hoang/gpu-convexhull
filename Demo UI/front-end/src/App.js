import React, {useEffect, useRef, useState} from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './index.css';
import MapboxGeocoder from '@mapbox/mapbox-gl-geocoder';
import '@mapbox/mapbox-gl-geocoder/dist/mapbox-gl-geocoder.css';
import * as turf from '@turf/turf';
import {featureCollection} from '@turf/turf';


mapboxgl.accessToken = 'pk.eyJ1IjoidGhhbmhoYWlpMDMiLCJhIjoiY2xwZ3R0dTJpMDFmODJxbGZpMTB0bG93dCJ9.zPguGUAukU-bfVTDnW-NlQ';


export default function App() {
    const mapContainer = useRef(null);
    const mapClickHandler = useRef(null);
    const map = useRef(null);
    var count = 0;
    var hull;
    const [centroid, setCentroid] = useState(featureCollection([]));
    const [markedDone, setMarkedDone] = useState(false);
    const [points, setPoints] = useState(featureCollection([]));


    
    useEffect(() => {
        if (map.current) return;
        map.current = new mapboxgl.Map({
            container: mapContainer.current,
            style: 'mapbox://styles/mapbox/streets-v12',
            center: [105.853333, 21.028333],
            zoom: 15,
        });
        map.current.on('load', async () => {
        });


        if (!markedDone) {
            mapClickHandler.current = handleMapClick;
            map.current.on('click', mapClickHandler.current);
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


    function getConvexHull() {
        hull = turf.convex(points);
        console.log(hull)
        console.log(hull.geometry.coordinates[0][1])

        const hullCoor = []
        for (let i=0; i<hull.geometry.coordinates.length; i++) {
            hullCoor[i] = [hull.geometry.coordinates[0][i]]
        }
        console.log(hullCoor)

        map.current.addLayer({
            id: `hull`,
            type: 'fill',
            source: {
                data: {
                    type: 'Feature',
                    geometry: {
                        type: 'Point',
                        coordinates: [hull.geometry.coordinates['0']]
                    }
                },
                type: 'geojson'
            },

            paint: {
                'fill-color': '#333',
            },
        });


    }

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
        // console.log(points.features.length);
        // console.log(points.features);

    }


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
        // Remove the previous event listener
        if (mapClickHandler.current) {
            map.current.off('click', mapClickHandler.current);
        }
        mapClickHandler.current = handleMapClick;
        map.current.on('click', mapClickHandler.current);

        count = 0;
    }

    function handleConfirm() {
        setMarkedDone(true);
        if (mapClickHandler.current) {
            map.current.off('click', mapClickHandler.current);
        }
        getConvexHull()
        getCentroid()


    }

    return (
        <div>
            <div className="sidebar">

                {!markedDone && (
                    <button onClick={handleConfirm}>Done</button>
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

                        <h2>The centroid: </h2>
                        <h3>{centroid.features["0"].geometry.coordinates[0]} , {centroid.features["0"].geometry.coordinates[0] } </h3>


                        <button onClick={handleReset}>Reset</button>
                    </div>
                )}
            </div>
            <div ref={mapContainer} className="map-container" />
        </div>
    );
}
