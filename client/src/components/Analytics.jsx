import React, { useState, useEffect } from "react";
import "./Analytics.css";

const Analytics = () => {
  const [analyticsData, setAnalyticsData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // API base URL
  const API_BASE = "http://127.0.0.1:8000";

  // Fetch analytics data from FastAPI backend
  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        // sample payload for /api/analytics
        const sampleData = {
          students: [
            {
              student: "Aarav",
              tasks_assigned: 10,
              tasks_completed: 9,
              commits: 25,
              feedback: "Aarav communicates well and delivers tasks on time.",
            },
            {
              student: "Priya",
              tasks_assigned: 8,
              tasks_completed: 6,
              commits: 15,
              feedback: "Priya needs to improve her pace, but great designs.",
            },
            {
              student: "Rohan",
              tasks_assigned: 7,
              tasks_completed: 7,
              commits: 18,
              feedback: "Rohan is consistent and helps others often.",
            },
            {
              student: "Aanya",
              tasks_assigned: 9,
              tasks_completed: 5,
              commits: 10,
              feedback: "Aanya missed some deadlines but contributes new ideas.",
            },
            {
              student: "Arjun",
              tasks_assigned: 6,
              tasks_completed: 4,
              commits: 12,
              feedback: "Arjun works hard but sometimes overthinks details.",
            },
          ],
        };

        // First try the real backend route
        const response = await fetch(`${API_BASE}/api/analytics`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(sampleData),
        });

        if (!response.ok) {
          // fallback to /analytics (demo mode)
          console.warn("⚠️ /api/analytics failed, trying /analytics instead...");
          const demoRes = await fetch(`${API_BASE}/analytics`);
          if (!demoRes.ok) throw new Error("Both analytics routes failed");
          const demoData = await demoRes.json();
          setAnalyticsData(demoData.students || demoData || []);
        } else {
          const data = await response.json();
          setAnalyticsData(data.students || []);
        }

        setLoading(false);
      } catch (err) {
        console.error("❌ Analytics Fetch Error:", err);
        setError(err.message);
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, []);

  // ----------------------------
  // Render logic
  // ----------------------------
  if (loading) {
    return (
      <div className="analytics-container">
        <h3>📊 Loading ML-powered Analytics...</h3>
      </div>
    );
  }

  if (error) {
    return (
      <div className="analytics-container">
        <h3>❌ Error loading analytics</h3>
        <p>{error}</p>
        <p>Make sure your FastAPI backend is running on port 8000.</p>
      </div>
    );
  }

  if (!analyticsData || analyticsData.length === 0) {
    return (
      <div className="empty-state-analytics">
        <div className="empty-icon-analytics">📊</div>
        <h3>No Analytics Data Yet</h3>
        <p>Backend returned no data. Try adding some student records.</p>
      </div>
    );
  }

  // ----------------------------
  // Render Table and Summary
  // ----------------------------
  return (
    <div className="analytics-container">
      <div className="analytics-header">
        <h2>📈 Project Analytics Dashboard</h2>
        <p>Real-time insights powered by Hugging Face Sentiment Analysis</p>
      </div>

      <div className="analytics-table-section">
        <table className="analytics-table">
          <thead>
            <tr>
              <th>Student</th>
              <th>Completion Rate (%)</th>
              <th>Commits</th>
              <th>Sentiment</th>
              <th>Sentiment Score</th>
              <th>Quality Score</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {analyticsData.map((row, index) => (
              <tr key={index}>
                <td>{row.student || row.Student}</td>
                <td>{row.completion_rate || row.Completion_Rate}</td>
                <td>{row.commits || row.Commits}</td>
                <td>
                  {row.sentiment_emoji
                    ? `${row.sentiment_emoji} ${row.sentiment}`
                    : row.sentiment}
                </td>
                <td>
                  {row.sentiment_score
                    ? row.sentiment_score.toFixed(2)
                    : row.Sentiment_Score}
                </td>
                <td>{row.quality_score || row.Performance_Score}</td>
                <td>{row.project_status || row.Project_Status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="analytics-summary">
        <h3>💡 Insights Summary</h3>
        <ul>
          <li>⚙️ Data includes automatic Hugging Face sentiment evaluation.</li>
          <li>
            📊 Completion rate, commits, and feedback are combined for
            AI-powered performance scoring.
          </li>
          <li>🚦 Status flags: 🟢 On Track | 🟠 Moderate | 🔴 At Risk</li>
        </ul>
      </div>
    </div>
  );
};

export default Analytics;
