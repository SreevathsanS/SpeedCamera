// src/components/PublicDashboard.jsx
import React, { useState, useEffect } from 'react';
import apiClient from '../api/apiClient';
import { useAuth } from '../context/AuthContext';
import styles from './PublicDashboard.module.css';

export default function PublicDashboard() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const { user } = useAuth();
  const API_URL = 'http://127.0.0.1:8000'; // Base URL for constructing file paths

  useEffect(() => {
    const fetchEvents = async () => {
      try {
        setLoading(true);
        const response = await apiClient.get('/api/events');
        setEvents(response.data);
      } catch (error) {
        console.error('Failed to fetch events:', error);
      } finally {
        setLoading(false);
      }
    };
    fetchEvents();
  }, []);

  if (loading) {
    return <p>Loading your challans...</p>;
  }

  return (
    <div className={styles.dashboard}>
      <h1>Challans for Vehicle: {user.username}</h1>
      {events.length > 0 ? (
        <div className={styles.challanGrid}>
          {events.map((event) => (
            <div key={event.id} className={styles.challanCard}>
              <img
                src={`${API_URL}${event.vehicle_image_url}`}
                alt={`Vehicle ${event.plate_number}`}
                className={styles.vehicleImage}
              />
              <div className={styles.cardContent}>
                <h3>Violation on {event.timestamp.split('_')[0]}</h3>
                <p>
                  <strong>Speed:</strong> {Math.round(event.speed_kmh)} km/h
                </p>
                <p>
                  <strong>Camera:</strong> {event.camera_id}
                </p>
                <a
                  href={`${API_URL}${event.challan_pdf_url}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={styles.button}
                >
                  View Challan (PDF)
                </a>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p>No challans found for your vehicle.</p>
      )}
    </div>
  );
}