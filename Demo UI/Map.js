
const key = 'sYblH2GAMFwVHjAJzIOG';

const attribution = new ol.control.Attribution({
    collapsible: false,
});

const source = new ol.source.TileJSON({
    url: `https://api.maptiler.com/maps/streets-v2/tiles.json?key=${key}`, // source URL
    tileSize: 512,
    crossOrigin: 'anonymous'
});

const map = new ol.Map({
    layers: [
        new ol.layer.Tile({
            source: source
        })
    ],
    controls: ol.control.defaults.defaults({attribution: false}).extend([attribution]),
    target: 'map',
    view: new ol.View({
        constrainResolution: true,
        // center: ol.proj.fromLonLat([105.804817,21.028511]), // starting position [lng, lat]
        center: ol.proj.fromLonLat([105.805313,21.038491]), // starting position [lng, lat]
        zoom: 15 // starting zoom
    })
});

