# Definimos una clase base llamada Vehiculo
class Vehiculo:
    def __init__(self, marca, modelo, color):
        self.marca = marca  # Atributo público
        self.modelo = modelo
        self.color = color
        self._velocidad = 0  # Atributo protegido (convención con _)

    def acelerar(self, incremento):
        """Método para aumentar la velocidad"""
        self._velocidad += incremento
        print(f"{self.marca} {self.modelo} aceleró a {self._velocidad} km/h")

    def frenar(self):
        """Método para detener el vehículo"""
        self._velocidad = 0
        print(f"{self.marca} {self.modelo} se ha detenido.")

    def mostrar_info(self):
        """Mostrar información del vehículo"""
        print(f"🚗 Vehículo: {self.marca} {self.modelo} ({self.color}) - Velocidad: {self._velocidad} km/h")


# Clase Coche que hereda de Vehiculo
class Coche(Vehiculo):
    def __init__(self, marca, modelo, color, puertas):
        super().__init__(marca, modelo, color)  # Llamamos al constructor de la clase padre
        self.puertas = puertas  # Atributo adicional

    def mostrar_info(self):
        """Sobrescribimos el método para incluir el número de puertas"""
        print(f"🚙 Coche: {self.marca} {self.modelo} ({self.color}) - {self.puertas} puertas - Velocidad: {self._velocidad} km/h")


# Clase Moto que hereda de Vehiculo
class Moto(Vehiculo):
    def __init__(self, marca, modelo, color, tipo):
        super().__init__(marca, modelo, color)
        self.tipo = tipo  # Ejemplo: deportiva, touring, naked

    def hacer_caballito(self):
        """Método exclusivo de las motos"""
        if self._velocidad > 20:
            print(f"🏍️ {self.marca} {self.modelo} está haciendo un caballito!")
        else:
            print(f"🏍️ {self.marca} {self.modelo} necesita más velocidad para hacer un caballito.")

    def mostrar_info(self):
        """Sobrescribimos el método para incluir el tipo de moto"""
        print(f"🏍️ Moto: {self.marca} {self.modelo} ({self.color}) - Tipo: {self.tipo} - Velocidad: {self._velocidad} km/h")

# Clase Camion que hereda de Vehiculo
class Camion(Vehiculo):
    def __init__(self, marca, modelo, color, carga_maxima):
        super().__init__(marca, modelo, color)
        self.carga_maxima = carga_maxima # Muestra la capacidad de carga

    def cargar_peso(self, peso):
        """Método exclusivo de los camiones"""
        if peso > self.carga_maxima:
            print(f"El peso ({peso} kg) supera la capacidad máxima permitida.")
        else:
            print(f"El peso ({peso} kg) no supera la capacidad máxima permitida.")
    
    def mostrar_info(self):
        """Sobrescribimos el método para incluir la carga máxima del camión"""
        print(f"🚛 Camión: {self.marca} {self.modelo} ({self.color}) - Carga máxima: {self.carga_maxima} kg.")
        
# --- PRUEBA DEL CÓDIGO ---
# Crear objetos
mi_coche = Coche("Toyota", "Corolla", "Rojo", 4)
mi_moto = Moto("Yamaha", "R6", "Azul", "Deportiva")
mi_camion = Camion("Ferrari", "3000", "Plateado", 1000)

# Usar métodos
mi_coche.mostrar_info()
mi_coche.acelerar(50)
mi_coche.frenar()

print("\n")  # Separador

mi_moto.mostrar_info()
mi_moto.acelerar(30)
mi_moto.hacer_caballito()
mi_moto.frenar()

print("\n")  # Separador

mi_camion.mostrar_info()
mi_camion.cargar_peso(1200)
mi_camion.acelerar(300000)
