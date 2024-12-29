const { MongoClient } = require('mongodb');
const username = 'bhaveshxop';
const password = '.E3uQP6F6U.pm2-';
const client = new MongoClient(`mongodb+srv://${username}:${password}@cluster0.1oaxi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0`);

module.exports = client