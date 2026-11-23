const uint8_t ANALOG_PIN = PA0;
const unsigned long SAMPLE_INTERVAL_US = 500; // 500 us = 2 kHz

unsigned long lastSampleMicros = 0;
uint16_t blinkCounter = 0;
bool ledState = false;

void setup() {
  pinMode(ANALOG_PIN, INPUT_ANALOG);
  pinMode(PC13, OUTPUT);
  analogReadResolution(12); // 0 to 4095 (0 V to 3.3 V at PA0)

  Serial1.begin(460800);    // PA9 (TX) / PA10 (RX)
  lastSampleMicros = micros();
}

void loop() {
  unsigned long now = micros();

  if (now - lastSampleMicros >= SAMPLE_INTERVAL_US) {
    if (now - lastSampleMicros > 5000) {
      lastSampleMicros = now;
    } else {
      lastSampleMicros += SAMPLE_INTERVAL_US;
    }

    uint16_t emgSample = analogRead(ANALOG_PIN);
    Serial1.println(emgSample);

    // Toggle onboard PC13 LED every 1000 samples (0.5 s ON / 0.5 s OFF)
    if (++blinkCounter >= 1000) {
      blinkCounter = 0;
      ledState = !ledState;
      digitalWrite(PC13, ledState ? LOW : HIGH);
    }
  }
}