const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');
const nodemailer = require('nodemailer');
const crypto = require('crypto');


const router = express.Router();
const JWT_SECRET = process.env.JWT_SECRET;

// Email transporter config (using Gmail as example — enable less secure or use OAuth2 in production)
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: { user: process.env.EMAIL, pass: process.env.EMAIL_PASS }
});

router.post('/register', async (req, res) => {
  const { fullName, email, department, year, password } = req.body;
  const hash = await bcrypt.hash(password, 10);
  const verifyToken = crypto.randomBytes(32).toString('hex');
  try {
    const user = await User.create({
      fullName, email, department, year,
      password: hash, verifyToken
    });
    const url = `http://localhost:3000/verify-email?token=${verifyToken}`;
    await transporter.sendMail({
      to: email,
      subject: 'Verify your email',
      html: `<a href="${url}">Click here to verify your email</a>`
    });
    res.status(201).json({ msg: "Check your email for verification link!" });
  } catch (err) {
    res.status(400).json({ error: 'Registration failed' });
  }
});

router.get('/verify-email', async (req, res) => {
  const { token } = req.query;
  const user = await User.findOne({ verifyToken: token });
  if (!user) {
    return res.status(400).send('Invalid token');
  }
  user.isVerified = true;
  user.verifyToken = undefined;
  await user.save();
  res.send('Email verified! You can now login.');
});

router.post('/login', async (req, res) => {
  const { email, password } = req.body;
  const user = await User.findOne({ email });
  if (!user)
    return res.status(400).json({ error: 'Invalid credentials' });
  if (!user.isVerified)
    return res.status(400).json({ error: 'Please verify your email' });
  const valid = await bcrypt.compare(password, user.password);
  if (!valid)
    return res.status(400).json({ error: 'Invalid credentials' });

  // JWT or session — here using JWT
  const token = jwt.sign({ id: user._id }, JWT_SECRET, { expiresIn: '1d' });
  res.json({ token, msg: "Login successful" });
});

module.exports = router;
