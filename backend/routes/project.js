const express = require("express");
const router = express.Router();
const jwt = require("jsonwebtoken");
const Project = require("../models/Project");

const JWT_SECRET = "mysecretkey123"; // same as in auth.js

// Middleware to check token
function authMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader) return res.status(401).json({ message: "No token" });

  const token = authHeader.split(" ")[1];
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch {
    return res.status(401).json({ message: "Invalid token" });
  }
}

// Create new project
router.post("/", authMiddleware, async (req, res) => {
  const { name } = req.body;
  try {
    const project = new Project({ name, userId: req.user.userId });
    await project.save();
    res.json(project);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Get all projects for logged-in user
router.get("/", authMiddleware, async (req, res) => {
  try {
    const projects = await Project.find({ userId: req.user.userId });
    res.json(projects);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Update project progress
router.put("/:id", authMiddleware, async (req, res) => {
  try {
    const { progress } = req.body;
    const project = await Project.findOneAndUpdate(
      { _id: req.params.id, userId: req.user.userId },
      { progress },
      { new: true }
    );
    res.json(project);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
