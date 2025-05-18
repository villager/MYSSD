import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.stats as stats

class KSImageCompareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparador de Imágenes KS")
        self.root.geometry("1200x800")
        
        # Variables para las imágenes (corregido typo en _gray)
        self.image1 = None
        self.image2 = None
        self.image1_gray = None
        self.image2_gray = None
        
        # Crear interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para controles
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Botones para cargar imágenes
        tk.Button(control_frame, text="Cargar Imagen 1", command=self.load_image1).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Cargar Imagen 2", command=self.load_image2).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Ejecutar Test KS", command=self.run_ks_test, 
                 bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Guardar Resultados", command=self.save_results,
                 bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Frame para mostrar imágenes y resultados
        display_frame = tk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame para imágenes originales
        img_frame = tk.Frame(display_frame)
        img_frame.pack(fill=tk.BOTH, expand=True)
        
        # Labels para imágenes con tamaño fijo
        self.label1 = tk.Label(img_frame, width=400, height=300)
        self.label1.pack(side=tk.LEFT, expand=True)
        
        self.label2 = tk.Label(img_frame, width=400, height=300)
        self.label2.pack(side=tk.LEFT, expand=True)
        
        # Frame para gráficos
        plot_frame = tk.Frame(display_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Gráfico de distribuciones acumulativas
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.fig.subplots_adjust(wspace=0.4)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Frame para resultados estadísticos
        result_frame = tk.Frame(display_frame)
        result_frame.pack(fill=tk.X, pady=10)
        
        # Tabla de resultados con scrollbar
        self.result_table = ttk.Treeview(result_frame, columns=("Metric", "Value"), show="headings", height=4)
        self.result_table.heading("Metric", text="Métrica")
        self.result_table.heading("Value", text="Valor")
        self.result_table.column("Metric", width=300, anchor='w')
        self.result_table.column("Value", width=300, anchor='w')
        
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_table.yview)
        self.result_table.configure(yscrollcommand=scrollbar.set)
        self.result_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Barra de estado
        self.status = tk.StringVar()
        self.status.set("Listo")
        tk.Label(main_frame, textvariable=self.status, bd=1, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X)
    
    def load_image1(self):
        path = filedialog.askopenfilename(
            title="Seleccionar primera imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif"), ("Todos los archivos", "*.*")]
        )
        if path:
            try:
                self.image1 = Image.open(path)
                self.image1_gray = self.image1.convert('L')
                self.display_image(self.image1, self.label1, "Imagen 1")
                self.status.set(f"Imagen 1 cargada: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")
    
    def load_image2(self):
        path = filedialog.askopenfilename(
            title="Seleccionar segunda imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif"), ("Todos los archivos", "*.*")]
        )
        if path:
            try:
                self.image2 = Image.open(path)
                self.image2_gray = self.image2.convert('L')
                self.display_image(self.image2, self.label2, "Imagen 2")
                self.status.set(f"Imagen 2 cargada: {path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {e}")
    
    def display_image(self, img, label, title):
        # Mantener aspect ratio al redimensionar
        img.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo
        label.config(text=title, compound=tk.TOP)
    
    def run_ks_test(self):
        if self.image1_gray is None or self.image2_gray is None:
            messagebox.showwarning("Advertencia", "Por favor cargue ambas imágenes primero")
            return
        
        try:
            # Obtener arrays de píxeles
            pixels1 = np.array(self.image1_gray).flatten()
            pixels2 = np.array(self.image2_gray).flatten()
            
            # Calcular histogramas normalizados
            hist1, bins = np.histogram(pixels1, bins=256, range=(0, 255), density=True)
            hist2, _ = np.histogram(pixels2, bins=256, range=(0, 255), density=True)
            
            # Calcular CDFs
            cdf1 = np.cumsum(hist1)
            cdf2 = np.cumsum(hist2)
            
            # Calcular estadístico KS
            ks_statistic, p_value = stats.ks_2samp(pixels1, pixels2)
            
            # Actualizar gráficos
            self.update_plots(cdf1, cdf2, bins)
            
            # Mostrar resultados
            self.show_results(ks_statistic, p_value)
            
            self.status.set(f"Test KS completado. p-value = {p_value:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error al procesar las imágenes: {e}")
    
    def update_plots(self, cdf1, cdf2, bins):
        # Limpiar gráficos
        self.ax1.clear()
        self.ax2.clear()
        
        # Gráfico de histogramas mejorado
        self.ax1.hist(np.array(self.image1_gray).flatten(), bins=256, range=(0, 255), 
                      alpha=0.5, density=True, label='Imagen 1', color='blue')
        self.ax1.hist(np.array(self.image2_gray).flatten(), bins=256, range=(0, 255), 
                      alpha=0.5, density=True, label='Imagen 2', color='orange')
        self.ax1.set_title('Histogramas de Intensidad')
        self.ax1.set_xlabel('Nivel de Gris')
        self.ax1.set_ylabel('Densidad de Probabilidad')
        self.ax1.legend()
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Gráfico de CDFs mejorado
        self.ax2.plot(bins[:-1], cdf1, label='Imagen 1', color='blue')
        self.ax2.plot(bins[:-1], cdf2, label='Imagen 2', color='orange')
        
        # Marcar la máxima diferencia
        diff = np.abs(cdf1 - cdf2)
        max_diff_idx = np.argmax(diff)
        self.ax2.axvline(bins[max_diff_idx], color='red', linestyle='--', alpha=0.5)
        self.ax2.text(bins[max_diff_idx], 0.5, f'Max Δ={diff[max_diff_idx]:.2f}', 
                     rotation=90, color='red')
        
        self.ax2.set_title('Funciones de Distribución Acumulativa')
        self.ax2.set_xlabel('Nivel de Gris')
        self.ax2.set_ylabel('Probabilidad Acumulada')
        self.ax2.legend()
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        
        self.canvas.draw()
    
    def show_results(self, ks_statistic, p_value):
        # Limpiar tabla de resultados
        for item in self.result_table.get_children():
            self.result_table.delete(item)

        # Agregar resultados
        self.result_table.insert("", "end", values=("Estadístico KS (D)", f"{ks_statistic:.4f}"))
        self.result_table.insert("", "end", values=("Valor p", f"{p_value:.4f}"))
        self.result_table.insert("", "end", values=("Nivel de significancia (α)", "0.05"))

        # Interpretación con umbral de significancia
        alpha = 0.05
        if p_value < alpha:
            conclusion = "CONCLUSIÓN: EXISTEN diferencias significativas (p < α)"
            color = "#ffdddd"  # Rojo claro
        else:
            conclusion = "CONCLUSIÓN: NO existen diferencias significativas (p ≥ α)"
            color = "#ddffdd"  # Verde claro

        # Añadir fila de conclusión
        self.result_table.insert("", "end", values=(conclusion, ""))
        self.result_table.tag_configure("conclusion", background=color)
        self.result_table.item(self.result_table.get_children()[-1], tags=("conclusion",))

        # Mostrar en la barra de estado
        self.status.set(conclusion + f" | D = {ks_statistic:.4f}, p = {p_value:.4f}")

    def save_results(self):
        if not hasattr(self, 'image1_gray') or not hasattr(self, 'image2_gray'):
            messagebox.showwarning("Advertencia", "No hay resultados para guardar")
            return

        output_path = filedialog.asksaveasfilename(
            title="Guardar resultados",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("Todos los archivos", "*.*")]
        )

        if output_path:
            try:
                # Crear figura para guardar
                save_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Histogramas
                ax1.hist(np.array(self.image1_gray).flatten(), bins=256, range=(0, 255), 
                        alpha=0.5, density=True, label='Imagen 1', color='blue')
                ax1.hist(np.array(self.image2_gray).flatten(), bins=256, range=(0, 255), 
                        alpha=0.5, density=True, label='Imagen 2', color='orange')
                ax1.set_title('Histogramas de Intensidad')
                ax1.set_xlabel('Nivel de Gris')
                ax1.set_ylabel('Densidad de Probabilidad')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # CDFs
                pixels1 = np.array(self.image1_gray).flatten()
                pixels2 = np.array(self.image2_gray).flatten()
                hist1, bins = np.histogram(pixels1, bins=256, range=(0, 255), density=True)
                hist2, _ = np.histogram(pixels2, bins=256, range=(0, 255), density=True)
                cdf1 = np.cumsum(hist1)
                cdf2 = np.cumsum(hist2)
                
                ax2.plot(bins[:-1], cdf1, label='Imagen 1', color='blue')
                ax2.plot(bins[:-1], cdf2, label='Imagen 2', color='orange')
                
                # Marcar máxima diferencia
                diff = np.abs(cdf1 - cdf2)
                max_diff_idx = np.argmax(diff)
                ax2.axvline(bins[max_diff_idx], color='red', linestyle='--', alpha=0.5)
                ax2.text(bins[max_diff_idx], 0.5, f'Max Δ={diff[max_diff_idx]:.2f}', 
                        rotation=90, color='red')
                
                ax2.set_title('Funciones de Distribución Acumulativa')
                ax2.set_xlabel('Nivel de Gris')
                ax2.set_ylabel('Probabilidad Acumulada')
                ax2.legend()
                ax2.grid(True, linestyle='--', alpha=0.7)
                
                # Resultados estadísticos
                ks_statistic, p_value = stats.ks_2samp(pixels1, pixels2)
                alpha = 0.05
                if p_value < alpha:
                    conclusion = "CONCLUSIÓN: Diferencias significativas (p < 0.05)"
                else:
                    conclusion = "CONCLUSIÓN: No hay diferencias significativas (p ≥ 0.05)"
                
                # Texto con formato mejorado
                plt.figtext(0.5, 0.01, 
                           f"Estadístico KS (D) = {ks_statistic:.4f}\nValor p = {p_value:.4f}\n{conclusion}", 
                           ha="center", fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
                
                save_fig.tight_layout(rect=[0, 0.05, 1, 0.95])
                save_fig.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close(save_fig)
                
                messagebox.showinfo("Éxito", f"Resultados guardados en:\n{output_path}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar los resultados: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = KSImageCompareApp(root)
    root.mainloop()