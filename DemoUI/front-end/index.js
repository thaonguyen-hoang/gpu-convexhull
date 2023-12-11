import React from 'react';
import { createRoot } from 'react-dom/client';
import 'mapbox-gl/dist/mapbox-gl.css';

import App from './App';

createRoot(document.getElementById('map')).render(
    <React.StrictMode>
        <App />
    </React.StrictMode>
);