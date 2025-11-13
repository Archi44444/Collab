const nodemailer = require('nodemailer');
require('dotenv').config();

const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: { user: process.env.EMAIL, pass: process.env.EMAIL_PASS }
});

transporter.sendMail({
  to: process.env.EMAIL,
  subject: 'Test email from CollabSphere',
  text: 'If you get this, Nodemailer is working!'
}, (err, info) => {
  if (err) console.error('Test mail error:', err);
  else console.log('Test mail success:', info.response);
});
console.log('EMAIL:', process.env.EMAIL);
console.log('EMAIL_PASS:', process.env.EMAIL_PASS ? 'present' : 'missing');
