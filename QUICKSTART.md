# 🚀 Quick Start Guide

## Get Started in 3 Steps

### 1. Clone and Setup
```bash
# If cloning from GitHub (when published)
git clone <your-repo-url>
cd hotel_booking_predictor

# Or if you have the folder locally
cd hotel_booking_predictor
```

### 2. Auto Setup (Recommended)
```bash
python run.py
```
This will:
- Check Python version
- Create virtual environment
- Install dependencies
- Extract and train the ML model
- Start the application

### 3. Manual Setup (Alternative)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Extract and train model
python extract_model.py

# Run the application
python app.py
```

## 🌐 Access the Application

Once running, open your browser and go to:
- **Main App**: http://localhost:5000
- **Prediction Form**: http://localhost:5000/predict
- **About Page**: http://localhost:5000/about
- **API Endpoint**: http://localhost:5000/api/predict

## 📱 Usage

1. Navigate to the **Predict** page
2. Fill in the booking details form
3. Click **"Predict Booking Status"**
4. View your prediction results with confidence score
5. Review insights and recommendations

## 🔧 Development

### Run in Development Mode
```bash
export FLASK_ENV=development  # Linux/macOS
set FLASK_ENV=development     # Windows
python app.py
```

### Run Tests
```bash
pip install pytest pytest-cov
pytest tests/
```

### Docker (Optional)
```bash
# Build and run with Docker
docker-compose up --build

# Or build manually
docker build -t hotel-booking-predictor .
docker run -p 5000:5000 hotel-booking-predictor
```

## 📊 Model Performance

Current model: **Random Forest**
- **Accuracy**: 87.20%
- **Features**: 14 booking characteristics
- **Prediction Time**: < 100ms

## 🆘 Troubleshooting

### Common Issues

**"Model artifacts not found"**
```bash
python extract_model.py
```

**"Dataset not found"**
- Ensure `first intern project.csv` is in the parent directory
- Or update the path in `extract_model.py`

**"Dependencies not installed"**
```bash
pip install -r requirements.txt
```

**"Port 5000 already in use"**
- Stop other applications using port 5000
- Or change the port in `app.py`: `app.run(port=5001)`

### Support

If you encounter issues:
1. Check the terminal output for error messages
2. Ensure all dependencies are installed
3. Verify the dataset file exists
4. Check Python version (3.8+ required)

## 🎯 Next Steps

- [ ] Customize the model by editing `extract_model.py`
- [ ] Add new features to the prediction form
- [ ] Implement user authentication
- [ ] Add data visualization dashboards
- [ ] Deploy to cloud platforms (AWS, Azure, GCP)
- [ ] Set up monitoring and logging

## 📚 Learn More

- Read the full [README.md](README.md) for detailed documentation
- Check the [About page](http://localhost:5000/about) in the app
- Explore the code in the `app/` directory
- Review the model training in `extract_model.py`

---

**Happy Predicting! 🏨📊**
