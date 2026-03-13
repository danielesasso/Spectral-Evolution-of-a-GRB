from ClassiPyGRB import SWIFT
import matplotlib.pyplot as plt


def test_classipy():
    print("Inizializzazione SWIFT...")
    swift = SWIFT(res=64)

    # scegliamo una GRB famosa
    grb_name = "GRB211211A"

    print(f"Scarico dati per {grb_name}...")
    df = swift.obtain_data(name=grb_name)

    print("\nPrime righe del dataset:")
    print(df.head())

    print("\nPlot della light curve...")
    swift.plot_any_grb(name=grb_name)
    plt.show()


if __name__ == "__main__":
    test_classipy()