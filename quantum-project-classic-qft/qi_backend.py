from __future__ import annotations

def get_tuna9_backend():
	try:
		from qiskit_quantuminspire.qi_provider import QIProvider
	except Exception as e:
		raise ImportError("Missing dependency: qiskit-quantuminspire. Install it and run `qi login`.") from e

	try:
		provider = QIProvider()
	except Exception as e:
		msg = str(e)
		if "ValidationError" in type(e).__name__ and "BackendType" in msg and "image_id" in msg:
			raise RuntimeError(
				"Quantum Inspire client/schema mismatch while fetching backend types. "
			) from e

		raise RuntimeError(
			"Could not initialize Quantum Inspire provider. Ensure you've run `qi login` and that "
			"qiskit-quantuminspire/quantuminspire dependencies are up-to-date."
		) from e

	for name in ("Tuna-9", "Tuna 9", "tuna-9", "tuna 9"):
		for call in (lambda: provider.get_backend(name), lambda: provider.get_backend(name=name)):
			try:
				return call()
			except Exception:
				pass

	available = [getattr(b, "name", str(b)) for b in provider.backends()]
	raise RuntimeError(f"Tuna-9 backend not found. Available backends: {available}")
