const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  fullName: String,
  email: { type: String, unique: true },
  department: String,
  year: String,
  password: String,
  isVerified: { type: Boolean, default: false },
  verifyToken: String,
});

module.exports = mongoose.model('User', userSchema);
