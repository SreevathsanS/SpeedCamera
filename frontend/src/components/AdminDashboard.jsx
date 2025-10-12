// src/components/AdminDashboard.jsx
import React, {useState, useEffect} from 'react';
import apiClient from '../api/apiClient';
import { useAuth } from '../context/AuthContext';
import styles from './AdminDashboard.module.css';

// Import the Document and Page components from react-pdf
import { Document, Page, pdfjs } from 'react-pdf';
// Import the styles for the viewer
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Configure the PDF.js worker to load from the public directory
pdfjs.GlobalWorkerOptions.workerSrc = `/pdf.worker.mjs`;

// --- Configuration ---
// Define your camera IDs and the base URL of your API server
const CAMERA_IDS = ['street_1', 'street_2'];
const API_URL = 'http://127.0.0.1:8000';

export default function AdminDashboard() {
  const [stats, setStats] = useState(null);
  const [events, setEvents] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  // --- ADDED STATE FOR PDF VIEWER ---
  const [activePdfUrl, setActivePdfUrl] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const { user } = useAuth(); // You may not need this for the admin dashboard, but it's good practice to keep if you ever need user context.

  useEffect(() => {
    // 1. Initial data fetch to load existing data
    const fetchData = async () => {
      try {
        setLoading(true);
        const [statsRes, eventsRes, alertsRes] = await Promise.all([
          apiClient.get('/api/stats'),
          apiClient.get('/api/events'),
          apiClient.get('/api/stolen-alerts'),
        ]);
        setStats(statsRes.data);
        setEvents(eventsRes.data);
        setAlerts(alertsRes.data.alerts);
        setError('');
      } catch (err) {
        setError('Failed to fetch initial dashboard data.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();

    // 2. Establish WebSocket connection for real-time updates
    const ws = new WebSocket('ws://127.0.0.1:8000/ws');

    ws.onopen = () => {
      console.log('WebSocket connection established');
    };

    // =======================================================
    // --- THIS IS THE CORRECTED SECTION ---
    // =======================================================
    ws.onmessage = (event) => {
      const newEventData = JSON.parse(event.data);
      console.log('Real-time event received:', newEventData);

      // Check the type of the event to handle it correctly
      if (newEventData.type === 'stolen_vehicle_alert') {
        // --- HANDLE STOLEN VEHICLE ALERT ---
        
        // Format the alert object into a readable string for the UI
        const formattedAlert = `${newEventData.timestamp}: STOLEN VEHICLE! Plate: ${newEventData.plate_number} seen on Camera: ${newEventData.camera_id}`;
        
        // Add the new alert to the top of the alerts list
        setAlerts(prevAlerts => [formattedAlert, ...prevAlerts]);

        // Also, update the stolen alerts stat card
        setStats(prevStats => {
            if (!prevStats) return null;
            return {
                ...prevStats,
                // Ensure stolen_alerts exists before incrementing
                stolen_alerts: (prevStats.stolen_alerts || 0) + 1,
            };
        });

      } else {
        // --- HANDLE OVERSPEEDING EVENT (default) ---

        // Add the new event to the top of the events list
        setEvents(prevEvents => [newEventData, ...prevEvents]);

        // Update the statistics in real-time
        setStats(prevStats => {
          if (!prevStats) return null;
          return {
            ...prevStats,
            total_violations: prevStats.total_violations + 1,
          };
        });
      }
    };
    // =======================================================
    // --- END OF CORRECTED SECTION ---
    // =======================================================

    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    // 3. Cleanup function: Close the connection when the component unmounts
    return () => {
      ws.close();
    };
  }, []); // The empty dependency array ensures this effect runs only once

  // --- ADDED HANDLERS FOR PDF VIEWER ---
  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };
  
  const handleViewChallan = (url) => {
    setActivePdfUrl(url);
    setPageNumber(1); // Reset to the first page when a new PDF is opened
  };

  if (loading) {
    return <p>Loading dashboard...</p>;
  }

  if (error) {
    return <p className={styles.error}>{error}</p>;
  }

  return (
    <div className={styles.dashboard}>
      <h1>Admin Dashboard</h1>

      {/* Live Video Feeds Section */}
      <h2>Live Camera Feeds</h2>
      <div className={styles.videoGrid}>
        {CAMERA_IDS.map(camId => (
          <div key={camId} className={styles.videoContainer}>
            <h3>{camId.replace('_', ' ').toUpperCase()}</h3>
            <img 
              src={`${API_URL}/api/video_feed/${camId}`}
              alt={`Live feed from ${camId}`}
              width="100%"
            />
          </div>
        ))}
      </div>

      {/* Statistics Section */}
      <h2>Overall Statistics</h2>
      <div className={styles.statsGrid}>
        <div className={styles.statCard}>
          <h2>{stats?.total_violations ?? 0}</h2>
          <p>Total Violations</p>
        </div>
        <div className={styles.statCard}>
          <h2>{stats?.unique_vehicles ?? 0}</h2>
          <p>Unique Vehicles</p>
        </div>
        <div className={styles.statCard}>
          <h2>{stats?.stolen_alerts ?? 0}</h2>
          <p>Stolen Vehicle Alerts</p>
        </div>
      </div>

      {/* Stolen Alerts & Recent Violations Section */}
      <div className={styles.columns}>
        <div className={styles.column}>
          <h2>Recent Violations</h2>
          <div className={styles.tableContainer}>
            <table className={styles.eventsTable}>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Plate</th>
                  <th>Speed</th>
                  <th>Camera</th>
                  <th>Challan</th>
                </tr>
              </thead>
              <tbody>
                {events.slice(0, 10).map((event, index) => (
                  <tr key={event.id || `new-${index}`}>
                    <td>{event.timestamp.replace('_', ' ')}</td>
                    <td>{event.plate_number}</td>
                    <td>{Math.round(event.speed_kmh)} km/h</td>
                    <td>{event.camera_id}</td>
                    <td>
                      {/* --- ADDED CHALLAN VIEW BUTTON --- */}
                      <button
                        onClick={() => handleViewChallan(`${API_URL}${event.challan_pdf_url}`)}
                        className={styles.viewChallanButton}
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className={styles.column}>
          <h2>Stolen Vehicle Alerts</h2>
          <ul className={styles.alertsList}>
            {alerts.length > 0 ? (
              alerts.slice(0, 10).map((alert, index) => (
                <li key={index}>{alert}</li>
              ))
            ) : (
              <li>No alerts found.</li>
            )}
          </ul>
        </div>
      </div>
      
      {/* --- ADDED PDF VIEWER SECTION --- */}
      {activePdfUrl && (
        <div className={styles.pdfViewerContainer}>
          <div className={styles.pdfControls}>
            <button
              onClick={() => setActivePdfUrl(null)}
              className={styles.closeButton}
            >
              Close Viewer
            </button>
            <button
              onClick={() => setPageNumber(prev => Math.max(prev - 1, 1))}
              disabled={pageNumber <= 1}
            >
              Previous Page
            </button>
            <span>
              Page {pageNumber} of {numPages}
            </span>
            <button
              onClick={() => setPageNumber(prev => Math.min(prev + 1, numPages))}
              disabled={pageNumber >= numPages}
            >
              Next Page
            </button>
          </div>
          <Document file={activePdfUrl} onLoadSuccess={onDocumentLoadSuccess}>
            <Page pageNumber={pageNumber} />
          </Document>
        </div>
      )}
    </div>
  );
}