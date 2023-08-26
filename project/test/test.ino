void setup(){
  pinMode(13,OUTPUT);
  Serial.begin(9600);
  int lcd = 12;
}
void loop(){
  digitalWrite(13,HIGH);
  delay(3000);
  digitalWrite(lcd,HIGH);
  delay(3000);
  digitalWrite(13,LOW);
  digitalWrite(lcd,LOW);
}
