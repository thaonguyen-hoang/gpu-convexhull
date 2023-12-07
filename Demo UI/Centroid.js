// import {getCenter as ol_extent_getCenter} from './node_modules/ol/extent.j'

var vector = new ol.layer.Vector({ source: new ol.source.Vector() })
map.addLayer(vector);

var draw = new ol.interaction.Draw({ source:vector.getSource(), type:"Point" });
map.addInteraction(draw);

var hull, pts;
draw.on("drawend", function(e){
  if (!vector.getSource().getFeatures().length) {
    hull = new ol.Feature(new ol.geom.Polygon([[0,0]]));
    vector.getSource().addFeature(hull);
    console.log(hull)
    pts = [];
  }
  pts.push(e.feature.getGeometry().getCoordinates());
  hull.setGeometry(new ol.geom.Polygon ( [ ol.coordinate.convexHull(pts) ] ));
});




// var ol_coordinate_getGeomCenter = function(geom) {
//       switch (geom.getType()) {
//         case 'Point':
//           return geom.getCoordinates();
//         case "MultiPolygon":
//           geom = geom.getPolygon(0);
//           // fallthrough
//         case "Polygon":
//           return geom.getInteriorPoint().getCoordinates();
//         default:
//           return geom.getClosestPoint(ol_extent_getCenter(geom.getExtent()));
//       }
//     };
  //
  //
  // console.log(hull)

  // var centroid = hull.getInteriorPoint().getCoordinates()
  // console.log(centroid)

  var centroidLayer = new ol.layer.Vector({
    source: new ol.source.Vector({
      features: [
        new ol.Feature({
          geometry : new ol.geom.Point(
            ol.proj.fromLonLat([105.805313,21.038491] )
          )
        })
      ]
    }),
    style : new ol.style.Style({
      image : new ol.style.Icon({
        src : "location-pin.png",
        scale : 0.05
      })
    })
  });

  map.addLayer(centroidLayer);