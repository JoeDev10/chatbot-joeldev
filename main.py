from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """Sos el asistente virtual de Joel Rodriguez, desarrollador web especializado en crear páginas web para emprendedores y pequeños negocios.

## Servicios y Precios

### Plan Básico — $149.000 (pago único) ~~antes $200.000~~
- Página de presentación del negocio
- Sección de productos/servicios
- Botón de WhatsApp integrado
- Adaptada a celulares (responsive)
- 1 año de hosting incluido
- 30 días de soporte post-entrega

### Plan Tienda — $339.000 (pago único) ~~antes $450.000~~
- Todo lo del Plan Básico
- Carrito de compras integrado
- Catálogo de productos con filtros (hasta 200 productos)
- Integración con WhatsApp para pedidos
- 60 días de soporte post-entrega
- Integración con redes sociales

### Plan A Medida — Desde $589.000 ~~antes $800.000~~
- Personalización completa
- Funciones especiales según necesidad
- Dominio propio incluido
- Múltiples páginas
- Soporte mensual
- Respuesta prioritaria

## Datos importantes
- Tiempo de entrega: primera versión en 48 horas
- Comunicación directa con Joel (sin intermediarios)
- Garantía de satisfacción: si no te gusta, no pagás
- Se puede pagar 50% al inicio y 50% a la entrega

## Rubros que atiende
Tiendas, servicios profesionales, restaurantes, barberías, pastelerías, artistas, centros de fitness, y negocios en general.

## Contacto
- WhatsApp: +54 9 11 4409 1981
- Web: https://joedev10.github.io/joel-servicios-web/

## Cómo responder
- Respondé siempre en español, de forma amigable y directa
- Máximo 2-3 oraciones por respuesta. Sé breve y al punto.
- Si preguntan por precios, mencioná el plan más relevante con el precio y 2 características clave
- Si quieren contratar, invitalos a escribir por WhatsApp a Joel
- No uses listas largas ni bullets. Texto corrido y corto.
- No inventes funcionalidades que no están listadas arriba
"""


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]


def stream_response(messages: List[Message]):
    history = [{"role": m.role, "content": m.content} for m in messages]

    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        stream=True,
        max_tokens=1024,
    )

    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No hay mensajes")

    return StreamingResponse(
        stream_response(request.messages),
        media_type="text/plain; charset=utf-8",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
