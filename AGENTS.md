# Contexto del proyecto

Este repositorio corresponde al Trabajo de Integración Curricular del proyecto:

"Desarrollo de una aplicación web para la predicción de demanda y simulación de recomendaciones de reposición de inventario mediante aprendizaje automático."

## Contexto del proyecto

- El proyecto utiliza datos públicos de ventas como caso de estudio.
- Corporación Favorita constituye únicamente el contexto asociado al conjunto de datos público utilizado.
- No existe integración con sistemas internos de Corporación Favorita ni acceso a datos privados.
- La solución está orientada a un usuario con funciones de análisis y planificación de inventario.

## Alcance de la predicción

- La granularidad considerada es tienda y familia de producto.
- Se trabajará con un conjunto acotado de series temporales representativas, seleccionado mediante criterios previamente establecidos.
- El horizonte de predicción es de 15 días.
- Se utilizará una línea base y un conjunto limitado de técnicas de predicción.
- Las técnicas concretas se seleccionarán durante la etapa experimental de acuerdo con las características de los datos y los resultados obtenidos.
- Se utilizarán implementaciones existentes de técnicas y bibliotecas de aprendizaje automático; no se desarrollarán algoritmos de aprendizaje automático desde cero.
- La validación debe respetar el orden temporal de los datos y evitar fuga de información futura.

## Simulación de inventario

El componente de inventario representa escenarios simulados de reposición.

No debe interpretarse como una recomendación operativa real ni como una integración con proveedores, sistemas ERP o sistemas internos de inventario.

## Forma de trabajo

- Priorizar soluciones simples, comprensibles y justificables académicamente.
- Mantener el alcance definido en el proyecto y evitar funcionalidades innecesarias.
- Antes de realizar cambios importantes, explicar brevemente qué se propone hacer y por qué.
- Cuando se implemente código relevante, explicar su funcionamiento y las decisiones técnicas para facilitar su comprensión y defensa académica.
- No introducir tecnologías o dependencias innecesarias sin justificar su utilidad para el proyecto.
- Mantener el código organizado, reproducible y fácil de probar.
- No modificar archivos ajenos a la tarea solicitada sin una razón clara.
- Antes de realizar cambios destructivos o difíciles de revertir, solicitar confirmación.