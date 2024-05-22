#include <Adafruit_GFX.h>    //OLED libraries
#include <Adafruit_SSD1306.h> //OLED libraries
#include "MAX30105.h"           //MAX3010x library
#include "heartRate.h"          //Heart rate calculating algorithm
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "health_model_data.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* ssid = "Neo Coffee";       // Tên WiFi của bạn
const char* password = "xincamon"; // Mật khẩu WiFi của bạn
const char* server = "http://192.168.2.32:8000/api/health-indexes"; // Địa chỉ server bạn muốn gửi dữ liệu

MAX30105 particleSensor;

const byte RATE_SIZE = 4; 
byte rates[RATE_SIZE]; 
byte rateSpot = 0;
float beatsPerMinute;
int beatAvg;

double avered = 0;
double aveir = 0;
double sumirrms = 0;
double sumredrms = 0;

double SpO2 = 0;
double ESpO2 = 90.0;
double FSpO2 = 0.7; 
double frate = 0.95; 
int i = 0;
int Num = 30;
#define FINGER_ON 7000    
#define MINIMUM_SPO2 90.0 

//OLED設定
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1 
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

static const unsigned char PROGMEM logo2_bmp[] =
{ 0x03, 0xC0, 0xF0, 0x06, 0x71, 0x8C, 0x0C, 0x1B, 0x06, 0x18, 0x0E, 0x02, 0x10, 0x0C, 0x03, 0x10,        
  0x04, 0x01, 0x10, 0x04, 0x01, 0x10, 0x40, 0x01, 0x10, 0x40, 0x01, 0x10, 0xC0, 0x03, 0x08, 0x88,
  0x02, 0x08, 0xB8, 0x04, 0xFF, 0x37, 0x08, 0x01, 0x30, 0x18, 0x01, 0x90, 0x30, 0x00, 0xC0, 0x60,
  0x00, 0x60, 0xC0, 0x00, 0x31, 0x80, 0x00, 0x1B, 0x00, 0x00, 0x0E, 0x00, 0x00, 0x04, 0x00,
};
static const unsigned char PROGMEM O2_bmp[] = {
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x00, 0x3f, 0xc3, 0xf8, 0x00, 0xff, 0xf3, 0xfc,
  0x03, 0xff, 0xff, 0xfe, 0x07, 0xff, 0xff, 0xfe, 0x0f, 0xff, 0xff, 0xfe, 0x0f, 0xff, 0xff, 0x7e,
  0x1f, 0x80, 0xff, 0xfc, 0x1f, 0x00, 0x7f, 0xb8, 0x3e, 0x3e, 0x3f, 0xb0, 0x3e, 0x3f, 0x3f, 0xc0,
  0x3e, 0x3f, 0x1f, 0xc0, 0x3e, 0x3f, 0x1f, 0xc0, 0x3e, 0x3f, 0x1f, 0xc0, 0x3e, 0x3e, 0x2f, 0xc0,
  0x3e, 0x3f, 0x0f, 0x80, 0x1f, 0x1c, 0x2f, 0x80, 0x1f, 0x80, 0xcf, 0x80, 0x1f, 0xe3, 0x9f, 0x00,
  0x0f, 0xff, 0x3f, 0x00, 0x07, 0xfe, 0xfe, 0x00, 0x0b, 0xfe, 0x0c, 0x00, 0x1d, 0xff, 0xf8, 0x00,
  0x1e, 0xff, 0xe0, 0x00, 0x1f, 0xff, 0x00, 0x00, 0x1f, 0xf0, 0x00, 0x00, 0x1f, 0xe0, 0x00, 0x00,
  0x0f, 0xe0, 0x00, 0x00, 0x07, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

long lastBeat = 0; // Khai báo biến lastBeat ở đây

boolean measurementStarted = false;
unsigned long startTime = 0;

// Các giá trị trung bình và độ lệch chuẩn từ mô hình trên Python
const float mean_pulse = 75.0;
const float std_pulse = 10.0;
const float mean_body_temperature = 36.5;
const float std_body_temperature = 0.5;
const float mean_SpO2 = 98.0;
const float std_SpO2 = 1.0;

const int numSamples = 119;

// Số lượng mẫu đã đọc
int samplesRead = numSamples;
// Các biến toàn cục được sử dụng cho TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// Dải tất cả các phép toán TFLM, bạn có thể loại bỏ dòng này và chỉ rút tất cả
// các phép toán TFLM bạn cần, nếu bạn muốn giảm kích thước được biên dịch của mã.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Tạo một bộ nhớ đệm tĩnh cho TFLM, kích thước có thể cần
// được điều chỉnh dựa trên mô hình bạn đang sử dụng
constexpr int tensorArenaSize = 16 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Mảng để ánh xạ chỉ số cử động sang tên
const char* GESTURES[] = {
  "0",
  "1",
  "2"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))


void setup() {
  Serial.begin(115200);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.display();
  WiFi.begin(ssid, password);
  
  // Kết nối WiFi
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  
  Serial.println("Connected to WiFi");

  delay(3000);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    while (1);
  }
  byte ledBrightness = 0x7F;
  byte sampleAverage = 4; 
  byte ledMode = 2; 
  int sampleRate = 800;
  int pulseWidth = 215;
  int adcRange = 16384; 
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  particleSensor.enableDIETEMPRDY();

  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);

  // Initialize TensorFlow Lite
  tflModel = tflite::GetModel(health_model_data);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Không khớp phiên bản của mô hình!");
    while (1);
  }

  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

// Khai báo biến toàn cục
unsigned long lastInferenceTime = 0;
const unsigned long inferenceInterval = 5000; // Thực hiện suy luận mỗi 5 giây
int sampleCount = 0;

// Buffer to store the samples
StaticJsonDocument<400> sampleBuffer;
JsonArray healthIndexes = sampleBuffer.to<JsonArray>();

void loop() {
  long irValue = particleSensor.getIR();
  if (irValue > FINGER_ON) {
    if (checkForBeat(irValue)) {
      delay(10);
      long delta = millis() - lastBeat;
      lastBeat = millis();
      beatsPerMinute = 60 / (delta / 1000.0);
      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        rates[rateSpot++] = (byte)beatsPerMinute;
        rateSpot %= RATE_SIZE;
        beatAvg = 0;
        for (byte x = 0; x < RATE_SIZE; x++) beatAvg += rates[x];
        beatAvg /= RATE_SIZE;
      }
    }

    uint32_t ir, red;
    double fred, fir;
    particleSensor.check();
    if (particleSensor.available()) {
      ir = particleSensor.getFIFOIR();
      red = particleSensor.getFIFORed();
      fir = (double)ir;
      fred = (double)red;
      aveir = aveir * frate + (double)ir * (1.0 - frate);
      avered = avered * frate + (double)red * (1.0 - frate);
      sumirrms += (fir - aveir) * (fir - aveir);
      sumredrms += (fred - avered) * (fred - avered);

      if ((i % Num) == 0) {
        double R = (sqrt(sumirrms) / aveir) / (sqrt(sumredrms) / avered);
        SpO2 = -23.3 * (R - 0.4) + 100;
        ESpO2 = FSpO2 * ESpO2 + (1.0 - FSpO2) * SpO2;
        if (ESpO2 <= MINIMUM_SPO2) ESpO2 = MINIMUM_SPO2;
        if (ESpO2 > 100) ESpO2 = 99.9;
        sumredrms = 0.0; sumirrms = 0.0; SpO2 = 0;
      }
      particleSensor.nextSample();
    }

    unsigned long now = millis();
    if (now - lastInferenceTime >= inferenceInterval) {
      lastInferenceTime = now;

      float ran = random(351, 361) / 10.0;

      float pulse = (int)(beatAvg / 1.5);
      if (pulse < 50) pulse = 50;
      float jsonSpO2 = ESpO2 + 5.1;
      if (jsonSpO2 > 100) jsonSpO2 = 99.9;
      float body_temperature = ran;

      pulse = (pulse - mean_pulse) / std_pulse;
      body_temperature = (body_temperature - mean_body_temperature) / std_body_temperature;
      float normalized_SpO2 = (jsonSpO2 - mean_SpO2) / std_SpO2;

      tflInputTensor->data.f[0] = pulse;
      tflInputTensor->data.f[1] = body_temperature;
      tflInputTensor->data.f[2] = normalized_SpO2;

      TfLiteStatus invokeStatus = tflInterpreter->Invoke();
      if (invokeStatus != kTfLiteOk) {
        Serial.println("Suy luận thất bại!");
        while (1);
        return;
      }

      int predicted_class = 0;
      float max_score = tflOutputTensor->data.f[0];
      for (int i = 1; i < NUM_GESTURES; ++i) {
        if (tflOutputTensor->data.f[i] > max_score) {
          max_score = tflOutputTensor->data.f[i];
          predicted_class = i;
        }
      }
      if (millis() - startTime >= 12000) {
        // Create a JSON object for the current sample
        StaticJsonDocument<200> jsonDocument;
        char spo2Str[10];
        char tempStr[10];
        dtostrf(jsonSpO2, 4, 2, spo2Str);
        dtostrf(ran, 4, 2, tempStr);
  
        int jsonPulse = (int)(beatAvg / 1.5);
        if (jsonPulse < 50) jsonPulse = 50;
  
        jsonDocument["pulse"] = jsonPulse;
        jsonDocument["spo2"] = spo2Str;
        jsonDocument["temperature"] = tempStr;
        jsonDocument["timeInMillis"] = now;
        jsonDocument["status"] = GESTURES[predicted_class];
        Serial.print("pulse: ");
        Serial.print(jsonPulse);
        Serial.print(", spo2: ");
        Serial.print(spo2Str);
        Serial.print(", temperature: ");
        Serial.print(tempStr);
        Serial.print(", time: ");
        Serial.print(now);
        Serial.print(", status: ");
        Serial.println(GESTURES[predicted_class]);
        // Add the sample to the buffer
        healthIndexes.add(jsonDocument);
  
        if (healthIndexes.size() >= 5) {  // Check if we have collected 5 samples
          if (WiFi.status() == WL_CONNECTED) {
            HTTPClient http;
            String jsonStr;
            serializeJson(sampleBuffer, jsonStr);
  
            http.begin(server);
            http.addHeader("Content-Type", "application/json");
            int httpResponseCode = http.POST(jsonStr);
            healthIndexes.clear();  // Clear the buffer after sending the data
          } else {
            Serial.println("WiFi Disconnected");
          }
        }
      
        // Update the OLED display with the same data
        display.clearDisplay();
        display.drawBitmap(5, 5, logo2_bmp, 24, 21, WHITE);
        display.setTextSize(2);
        display.setTextColor(WHITE);
        display.setCursor(42, 10);
        display.print(jsonPulse); 
        display.println(" BPM");
        display.drawBitmap(0, 35, O2_bmp, 32, 32, WHITE);
        display.setCursor(42, 40);
        display.print(String(spo2Str) + "%");
        display.display();
    }
    }
  } else {
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(WHITE);
    display.setCursor(30, 5);
    display.println("Finger");
    display.setCursor(30, 35);
    display.println("Please");
    display.display();
    delay(5000);
  }
}