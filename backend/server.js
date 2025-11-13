const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const app = express();

app.use(cors());
app.use(express.json());

// Connect MongoDB
mongoose.connect("mongodb://localhost:27017/todoApp")
  .then(() => console.log("✅ MongoDB connected"))
  .catch(err => console.error(err));

// Routes
const authRoutes = require("./routes/auth");
const projectRoutes = require("./routes/project");

app.use("/api/auth", authRoutes);
app.use("/api/projects", projectRoutes);

app.listen(5000, () => console.log("🚀 Server running on http://localhost:5000"));
