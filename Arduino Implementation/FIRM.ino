int sensorpin = 0;
int micaval = A0;
int micbval = A1;
int miccval = A2;
unsigned long timer;
unsigned long timer2;
unsigned long timer3;

void setup() {
  ADCSRA &= ~(bit (ADPS0) | bit (ADPS1) | bit (ADPS2));
  ADCSRA |= bit (ADPS1);
  Serial.begin(9600);
}

void loop() {
  int micaval = analogRead(A0);
  int micbval = analogRead(A1);
  int miccval = analogRead(A2);
  if(miccval > 360)
  {
    timer3 = micros();
    Serial.print(miccval);
    Serial.print(" ");
    Serial.println(timer3+3);
  }
  if(micbval > 360)
  {
    timer2 = micros();
    Serial.print(micbval);
    Serial.print(" ");
    Serial.println(timer2+2);
  }
  if(micaval > 360)
  {
    timer = micros();
    Serial.print(micaval);
    Serial.print(" ");
    Serial.println(timer+1);
  }
}
